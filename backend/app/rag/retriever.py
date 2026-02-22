"""
Adaptive hybrid retrieval combining FAISS semantic search and BM25 keyword search,
with MMR-lite diversity control.

Key design decisions:
- Candidate pool is the UNION of semantic top-N and BM25 top-N, ensuring
  neither signal can fully exclude the other.
- Coverage score reflects both peak similarity and mean support strength
  across the selected set, not just the top-1 score.
- Keyword density threshold is calibrated to the actual [0.0, 1.0] range
  produced by query_analyzer (domain-term fraction), not lexical diversity.
- Chunk embeddings are cached on the RetrievalResult so the evaluator can
  reuse them without a second encode call.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from numpy.typing import NDArray
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.router.router import Classification

logger = logging.getLogger(__name__)

MAX_CHUNKS_PER_DOC = 2

# Keyword density threshold calibrated to the domain-term fraction metric.
# Values above this indicate the query is term-heavy and BM25 should be
# weighted higher (e.g., "saml oauth sso configuration" → ~0.75).
_HIGH_DENSITY_THRESHOLD = 0.20


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    section_path: str
    page_number: int
    doc_type: str
    text: str
    token_count: int
    semantic_score: float
    bm25_score: float
    fusion_score: float


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    coverage_score: float
    spread_score: float
    top_k: int
    # Pre-computed embeddings of selected chunk texts, shaped (n_chunks, dim).
    # Passed to the evaluator to avoid re-encoding the same texts.
    chunk_embeddings: NDArray[np.float32] = field(
        default_factory=lambda: np.empty((0,), dtype=np.float32)
    )


class HybridRetriever:
    def __init__(self) -> None:
        self._model: Optional[SentenceTransformer] = None
        self._faiss_index: Optional[faiss.Index] = None
        self._bm25: Optional[BM25Okapi] = None
        self._metadata: list[dict] = []
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return

        logger.info("Loading embedding model: %s", settings.embedding_model)
        self._model = SentenceTransformer(settings.embedding_model)

        faiss_path = Path(settings.faiss_index_path)
        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
        self._faiss_index = faiss.read_index(str(faiss_path))

        bm25_path = Path(settings.bm25_corpus_path)
        if not bm25_path.exists():
            raise FileNotFoundError(f"BM25 corpus not found: {bm25_path}")
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]

        meta_path = Path(settings.metadata_path)
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
        with open(meta_path) as f:
            self._metadata = json.load(f)

        self._loaded = True
        logger.info("Retriever loaded (%d chunks indexed)", len(self._metadata))

    def _top_k_for_classification(self, classification: Classification) -> int:
        if classification == Classification.SIMPLE:
            return settings.top_k_simple
        return settings.top_k_complex

    def _fusion_weights(self, keyword_density: float) -> tuple[float, float]:
        if keyword_density >= _HIGH_DENSITY_THRESHOLD:
            return settings.w_sem_high_density, settings.w_bm25_high_density
        return settings.w_sem_default, settings.w_bm25_default

    def retrieve(
        self,
        query: str,
        classification: Classification,
        keyword_density: float,
        top_k_override: Optional[int] = None,
    ) -> RetrievalResult:
        self._load()

        top_k = top_k_override or self._top_k_for_classification(classification)
        # Fetch a larger candidate pool so diversity filtering has room to work
        fetch_k = min(top_k * 5, len(self._metadata))

        assert self._model is not None
        assert self._faiss_index is not None
        assert self._bm25 is not None

        query_embedding = self._model.encode(
         [query],
         normalize_embeddings=True,
         batch_size=32,
         show_progress_bar=False,
        ).astype(np.float32)

         # Semantic retrieval
        sem_scores_raw, sem_indices_raw = self._faiss_index.search(query_embedding, fetch_k)
        sem_scores_arr = sem_scores_raw[0]
        sem_indices_arr = sem_indices_raw[0]

        # BM25 retrieval — full corpus scores, normalized to [0, 1]
        tokenized_query = query.lower().split()
        bm25_raw: NDArray[np.float64] = self._bm25.get_scores(tokenized_query)
        bm25_max = float(bm25_raw.max()) if bm25_raw.max() > 0 else 1.0
        bm25_normalized = bm25_raw / bm25_max

        # Build BM25 top-N indices for union pool
        bm25_top_indices = set(
            np.argsort(bm25_raw)[::-1][:fetch_k].tolist()
        )
        sem_top_indices = set(
            int(idx) for idx in sem_indices_arr if idx >= 0 and idx < len(self._metadata)
        )
        candidate_indices = sem_top_indices | bm25_top_indices

        # Build a per-index score lookup from the FAISS results
        sem_score_by_idx: dict[int, float] = {
            int(sem_indices_arr[r]): float(sem_scores_arr[r])
            for r in range(len(sem_indices_arr))
            if sem_indices_arr[r] >= 0 and sem_indices_arr[r] < len(self._metadata)
        }

        w_sem, w_bm25 = self._fusion_weights(keyword_density)

        candidates: list[RetrievedChunk] = []
        for idx in candidate_indices:
            meta = self._metadata[idx]
            sem_score = sem_score_by_idx.get(idx, 0.0)
            bm25_score = float(bm25_normalized[idx])
            fusion_score = w_sem * sem_score + w_bm25 * bm25_score
            candidates.append(
                RetrievedChunk(
                    chunk_id=meta["chunk_id"],
                    doc_id=meta["doc_id"],
                    section_path=meta["section_path"],
                    page_number=meta["page_number"],
                    doc_type=meta["doc_type"],
                    text=meta["text"],
                    token_count=meta["token_count"],
                    semantic_score=sem_score,
                    bm25_score=bm25_score,
                    fusion_score=fusion_score,
                )
            )

        candidates.sort(key=lambda c: c.fusion_score, reverse=True)
        selected = self._apply_diversity_filter(candidates, top_k)

        coverage_score = self._compute_coverage(selected)
        spread_score = self._compute_spread(selected)

        # Cache chunk embeddings for the evaluator
        chunk_embeddings: NDArray[np.float32] = np.empty((0,), dtype=np.float32)
        if selected:
            chunk_embeddings = self._model.encode(
                [c.text for c in selected],
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False,
            ).astype(np.float32)

        return RetrievalResult(
            chunks=selected,
            coverage_score=coverage_score,
            spread_score=spread_score,
            top_k=top_k,
            chunk_embeddings=chunk_embeddings,
        )

    def _apply_diversity_filter(
        self, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        selected: list[RetrievedChunk] = []
        doc_counts: dict[str, int] = {}
        seen_section_paths: set[str] = set()

        for chunk in candidates:
            if len(selected) >= top_k:
                break
            if doc_counts.get(chunk.doc_id, 0) >= MAX_CHUNKS_PER_DOC:
                continue
            if chunk.section_path in seen_section_paths:
                continue
            selected.append(chunk)
            doc_counts[chunk.doc_id] = doc_counts.get(chunk.doc_id, 0) + 1
            seen_section_paths.add(chunk.section_path)

        # Relax section-path constraint if we haven't filled top_k
        if len(selected) < top_k:
            selected_ids = {c.chunk_id for c in selected}
            for chunk in candidates:
                if chunk.chunk_id in selected_ids:
                    continue
                if doc_counts.get(chunk.doc_id, 0) >= MAX_CHUNKS_PER_DOC:
                    continue
                selected.append(chunk)
                doc_counts[chunk.doc_id] = doc_counts.get(chunk.doc_id, 0) + 1
                selected_ids.add(chunk.chunk_id)
                if len(selected) >= top_k:
                    break

        return selected

    def _compute_coverage(self, chunks: list[RetrievedChunk]) -> float:
        """
        Coverage reflects both the strength of the best-matching chunk (peak
        relevance) and the mean support across the top-3 chunks.

        coverage = 0.7 * max_semantic_score + 0.3 * mean(top_3_semantic_scores)

        Using top-3 rather than the full selected set prevents tail-noise chunks
        (low-similarity, admitted only to fill k) from diluting the score.
        """
        if not chunks:
            return 0.0
        scores = sorted((c.semantic_score for c in chunks), reverse=True)
        max_score = scores[0]
        top3_mean = float(np.mean(scores[:3]))
        return 0.7 * max_score + 0.3 * top3_mean

    def _compute_spread(self, chunks: list[RetrievedChunk]) -> float:
        """
        Semantic spread: entropy of the doc_id distribution across selected
        chunks.  High entropy → evidence drawn from diverse documents.
        Low entropy → evidence concentrated in one document.
        """
        if len(chunks) < 2:
            return 0.0
        doc_ids = [c.doc_id for c in chunks]
        unique_docs = set(doc_ids)
        if len(unique_docs) == 1:
            return 0.0
        n = len(doc_ids)
        entropy = 0.0
        for doc in unique_docs:
            p = doc_ids.count(doc) / n
            entropy -= p * np.log2(p)
        # Normalize to [0, 1] by log2(n_unique)
        return float(entropy / np.log2(len(unique_docs)))


retriever = HybridRetriever()