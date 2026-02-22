"""
Offline document ingestion pipeline.

Usage:
    python -m backend.scripts.ingest --docs-dir docs/ --output-dir backend/data/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

TOKENIZER = tiktoken.get_encoding("cl100k_base")
TARGET_TOKENS = 450
OVERLAP_TOKENS = 75
HARD_MAX_TOKENS = 600
MIN_SEGMENT_TOKENS = 40
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Structural detection patterns — all are actively used in segmentation.
# ---------------------------------------------------------------------------

_HEADING_MD = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_HEADING_UNDERLINE = re.compile(r"^(.+)\n[=\-]{3,}\s*$", re.MULTILINE)
_NUMBERED_SECTION = re.compile(r"^(\d+(?:\.\d+)*[\.\)])\s+\S", re.MULTILINE)
_BULLET_CLUSTER = re.compile(r"(?:(?:^[ \t]*[-\*\+\u2022]\s+.+$)\n?){2,}", re.MULTILINE)
_CODE_BLOCK = re.compile(r"```[^\n]*\n[\s\S]*?```", re.MULTILINE)
_TABLE_BLOCK = re.compile(r"(?:^\|.+\|[ \t]*$\n?){2,}", re.MULTILINE)
_PAGE_MARKER = re.compile(r"\x00PAGE(\d+)\x00")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    section_path: str
    page_number: int
    doc_type: str
    text: str
    token_count: int


@dataclass
class Segment:
    text: str
    seg_type: str
    atomic: bool = False


class SegType:
    CODE = "code"
    TABLE = "table"
    BULLET = "bullet"
    PLAIN = "plain"


# ---------------------------------------------------------------------------
# Token utilities
# ---------------------------------------------------------------------------


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


def token_split_with_overlap(text: str, target: int, overlap: int) -> list[str]:
    tokens = TOKENIZER.encode(text)
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + target, len(tokens))
        chunks.append(TOKENIZER.decode(tokens[start:end]))
        if end >= len(tokens):
            break
        start = end - overlap
    return chunks


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------


def extract_text_pymupdf(pdf_path: Path) -> list[tuple[int, str]]:
    import fitz

    pages: list[tuple[int, str]] = []
    with fitz.open(str(pdf_path)) as doc:
        for i, page in enumerate(doc):
            pages.append((i + 1, page.get_text("text")))
    return pages


def extract_text_pdfplumber(pdf_path: Path) -> list[tuple[int, str]]:
    import pdfplumber

    pages: list[tuple[int, str]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append((i + 1, text))
    return pages


def extract_text(pdf_path: Path) -> list[tuple[int, str]]:
    try:
        pages = extract_text_pymupdf(pdf_path)
        if all(not t.strip() for _, t in pages):
            raise ValueError("PyMuPDF produced empty output")
        return pages
    except Exception as exc:
        logger.warning("PyMuPDF failed (%s), falling back to pdfplumber", exc)
        return extract_text_pdfplumber(pdf_path)


# ---------------------------------------------------------------------------
# Structure-aware segmentation
# ---------------------------------------------------------------------------


def _collect_atomic_spans(text: str) -> list[tuple[int, int, str]]:
    """
    Find all code block and table spans, returning non-overlapping sorted spans.
    Code blocks take priority over tables when ranges overlap.
    """
    spans: list[tuple[int, int, str]] = []

    for m in _CODE_BLOCK.finditer(text):
        spans.append((m.start(), m.end(), SegType.CODE))

    for m in _TABLE_BLOCK.finditer(text):
        overlaps = any(s <= m.start() < e or s < m.end() <= e for s, e, _ in spans)
        if not overlaps:
            spans.append((m.start(), m.end(), SegType.TABLE))

    spans.sort(key=lambda x: x[0])
    return spans


def _interleave_atomic_and_prose(text: str) -> list[Segment]:
    """
    Split text into an ordered list of atomic segments (code/table) and prose
    segments between them.  Atomic segments are returned verbatim; prose
    segments are passed to _split_prose_on_structure.
    """
    spans = _collect_atomic_spans(text)
    segments: list[Segment] = []
    cursor = 0

    for start, end, seg_type in spans:
        if cursor < start:
            prose = text[cursor:start]
            if prose.strip():
                segments.extend(_split_prose_on_structure(prose))
        block = text[start:end].strip()
        if block:
            segments.append(Segment(text=block, seg_type=seg_type, atomic=True))
        cursor = end

    tail = text[cursor:]
    if tail.strip():
        segments.extend(_split_prose_on_structure(tail))

    return segments


def _split_prose_on_structure(text: str) -> list[Segment]:
    """
    Split prose text on heading and numbered-section boundaries.
    Within each resulting slice, detect and tag bullet clusters.
    """
    split_positions: list[int] = []
    for m in _NUMBERED_SECTION.finditer(text):
        split_positions.append(m.start())
    for m in _HEADING_MD.finditer(text):
        split_positions.append(m.start())
    for m in _HEADING_UNDERLINE.finditer(text):
        split_positions.append(m.start())

    split_positions = sorted(set(split_positions))

    if not split_positions:
        return _tag_bullet_clusters(text)

    slices: list[str] = []
    prev = 0
    for pos in split_positions:
        if pos > prev:
            slices.append(text[prev:pos])
        prev = pos
    slices.append(text[prev:])

    result: list[Segment] = []
    for sl in slices:
        if sl.strip():
            result.extend(_tag_bullet_clusters(sl))
    return result


def _tag_bullet_clusters(text: str) -> list[Segment]:
    """
    Within a prose slice, extract bullet clusters as atomic units and return
    the surrounding text as plain segments.
    """
    spans = [(m.start(), m.end()) for m in _BULLET_CLUSTER.finditer(text)]
    if not spans:
        return [Segment(text=text, seg_type=SegType.PLAIN, atomic=False)]

    segments: list[Segment] = []
    cursor = 0
    for start, end in spans:
        if cursor < start:
            prose = text[cursor:start]
            if prose.strip():
                segments.append(Segment(text=prose, seg_type=SegType.PLAIN, atomic=False))
        block = text[start:end].strip()
        if block:
            segments.append(Segment(text=block, seg_type=SegType.BULLET, atomic=True))
        cursor = end
    tail = text[cursor:]
    if tail.strip():
        segments.append(Segment(text=tail, seg_type=SegType.PLAIN, atomic=False))
    return segments


def segment_text(text: str) -> list[Segment]:
    return _interleave_atomic_and_prose(text)


def merge_small_segments(segments: list[Segment], min_tokens: int) -> list[Segment]:
    """
    Forward-merge segments below min_tokens into their successor to avoid
    orphan chunks.  Atomic segments are always preserved as-is.
    """
    if not segments:
        return segments

    merged: list[Segment] = []
    pending: Segment | None = None

    for seg in segments:
        if pending is None:
            pending = seg
            continue
        if not pending.atomic and not seg.atomic and count_tokens(pending.text) < min_tokens:
            combined = pending.text.rstrip() + "\n\n" + seg.text.lstrip()
            pending = Segment(text=combined, seg_type=pending.seg_type, atomic=False)
        else:
            merged.append(pending)
            pending = seg

    if pending is not None:
        merged.append(pending)

    return merged


# ---------------------------------------------------------------------------
# Section path inference
# ---------------------------------------------------------------------------


def infer_section_path(text: str, doc_id: str, index: int) -> str:
    m = _HEADING_MD.search(text)
    if m:
        return m.group(2).strip()[:120]
    m = _HEADING_UNDERLINE.search(text)
    if m:
        return m.group(1).strip()[:120]
    m = _NUMBERED_SECTION.search(text)
    if m:
        line_end = text.find("\n", m.start())
        line = text[m.start(): line_end if line_end != -1 else m.start() + 80].strip()
        return line[:120]
    return f"{doc_id}/section_{index}"


# ---------------------------------------------------------------------------
# Document chunking with cross-page context
# ---------------------------------------------------------------------------


def chunk_document(
    doc_id: str, pages: list[tuple[int, str]], doc_type: str
) -> list[Chunk]:
    """
    Concatenate all pages into a single string before segmenting to preserve
    structural context across page boundaries.  Page markers embedded in the
    string let us track provenance without splitting prematurely.
    """
    page_parts: list[str] = []
    for page_num, text in pages:
        if text.strip():
            page_parts.append(f"\x00PAGE{page_num}\x00\n{text}")
    full_doc = "\n\n".join(page_parts)

    segments = segment_text(full_doc)
    segments = merge_small_segments(segments, MIN_SEGMENT_TOKENS)

    chunks: list[Chunk] = []
    chunk_index = 0

    for seg in segments:
        clean_text = _PAGE_MARKER.sub("", seg.text).strip()
        if not clean_text:
            continue

        first_marker = _PAGE_MARKER.search(seg.text)
        page_num = int(first_marker.group(1)) if first_marker else pages[0][0]

        tc = count_tokens(clean_text)
        if tc == 0:
            continue

        if seg.atomic or tc <= HARD_MAX_TOKENS:
            sub_texts = [clean_text]
        else:
            sub_texts = token_split_with_overlap(clean_text, TARGET_TOKENS, OVERLAP_TOKENS)

        for sub in sub_texts:
            sub = sub.strip()
            if not sub:
                continue
            tc_sub = count_tokens(sub)
            if tc_sub == 0:
                continue

            section_path = infer_section_path(sub, doc_id, chunk_index)
            chunk_id = hashlib.sha256(
                f"{doc_id}:{page_num}:{chunk_index}:{sub[:64]}".encode()
            ).hexdigest()[:16]

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    section_path=section_path,
                    page_number=page_num,
                    doc_type=doc_type,
                    text=sub,
                    token_count=tc_sub,
                )
            )
            chunk_index += 1

    return chunks


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------


def build_index(chunks: list[Chunk], output_dir: Path, embedding_model: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading embedding model: %s", embedding_model)
    model = SentenceTransformer(embedding_model)

    texts = [c.text for c in chunks]
    logger.info("Encoding %d chunks", len(texts))
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(output_dir / "faiss.index"))
    logger.info("FAISS index written (%d vectors, dim=%d)", index.ntotal, dim)

    tokenized_corpus = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(output_dir / "bm25_corpus.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "corpus": tokenized_corpus}, f)
    logger.info("BM25 corpus written")

    metadata = [asdict(c) for c in chunks]
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata written (%d chunks)", len(metadata))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def ingest(docs_dir: Path, output_dir: Path, embedding_model: str) -> None:
    pdf_files = sorted(docs_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", docs_dir)
        return

    all_chunks: list[Chunk] = []
    for pdf_path in pdf_files:
        doc_id = pdf_path.stem
        logger.info("Processing: %s", pdf_path.name)
        pages = extract_text(pdf_path)
        chunks = chunk_document(doc_id, pages, doc_type="pdf")
        logger.info("  -> %d chunks from %s", len(chunks), pdf_path.name)
        all_chunks.extend(chunks)

    logger.info("Total chunks: %d", len(all_chunks))
    build_index(all_chunks, output_dir, embedding_model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clearpath document ingestion")
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--output-dir", type=Path, default=Path("backend/data"))
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    args = parser.parse_args()
    ingest(args.docs_dir, args.output_dir, args.embedding_model)


if __name__ == "__main__":
    main()