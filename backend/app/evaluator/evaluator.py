"""
Advanced output evaluator with sentence-level attribution scoring,
refusal detection, domain entity guard, and prompt leakage detection.

Performance note: chunk embeddings are accepted as a pre-computed parameter
(produced by the retriever) to avoid a redundant encode call per request.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.rag.retriever import RetrievedChunk

_model: Optional[SentenceTransformer] = None

# Known Clearpath-domain entities. Presence in the answer is a positive signal.
CLEARPATH_ENTITIES: frozenset[str] = frozenset(
    [
        "clearpath",
        "dashboard",
        "workspace",
        "pipeline",
        "integration",
        "api key",
        "webhook",
        "connector",
        "data source",
        "report",
        "automation",
        "user role",
        "billing",
        "subscription",
        "team",
        "organization",
        "audit log",
        "access control",
        "sso",
        "saml",
        "oauth",
        "export",
        "import",
        "notification",
        "alert",
        "schedule",
    ]
)

# Competitor or out-of-domain SaaS products.  Presence in the answer when
# not mentioned in the retrieved context is a hallucination risk signal.
_COMPETITOR_ENTITIES: frozenset[str] = frozenset(
    [
        "jira",
        "asana",
        "monday.com",
        "notion",
        "trello",
        "clickup",
        "airtable",
        "smartsheet",
        "basecamp",
        "linear",
        "confluence",
        "sharepoint",
        "zendesk",
        "freshdesk",
        "intercom",
        "hubspot",
        "salesforce",
        "zapier",
        "make.com",
        "n8n",
        "fivetran",
        "airbyte",
        "stitch",
        "talend",
        "informatica",
        "databricks",
        "snowflake",
        "dbt",
        "looker",
        "tableau",
        "power bi",
        "metabase",
    ]
)

REFUSAL_PATTERNS: list[re.Pattern] = [
    re.compile(r"I (cannot|can't|am unable to|won't) (help|assist|answer)", re.I),
    re.compile(r"(outside|beyond) (my|the) (scope|capabilities|knowledge)", re.I),
    re.compile(r"I don'?t (have|possess) (access|information)", re.I),
    re.compile(r"as an AI (language model|assistant)", re.I),
    # Contract-required refusal phrases
    re.compile(r"not\s+mentioned\s+in\s+(the\s+)?(context|documentation|docs)", re.I),
    re.compile(r"cannot\s+find\s+(this\s+)?(information|answer|detail)", re.I),
    re.compile(r"no\s+information\s+(available|found|provided)\s+(about|on|for)", re.I),
]

PROMPT_LEAKAGE_PATTERNS: list[re.Pattern] = [
    re.compile(r"<retrieved_context>", re.I),
    re.compile(r"IMPORTANT SECURITY RULES", re.I),
    re.compile(r"system\s+prompt\s+(prefix|content|says)", re.I),
    re.compile(r"you are a helpful customer support assistant", re.I),
]

MIN_DISTINCT_CHUNKS = 2
_ATTRIBUTION_SUPPORT_THRESHOLD = 0.40


class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class EvaluationResult:
    confidence: Confidence
    flags: list[str]
    attribution_mean: float
    attribution_min: float
    attribution_variance: float
    distinct_supporting_chunks: int
    has_domain_entities: bool
    is_refusal: bool
    has_prompt_leakage: bool


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if len(s.split()) >= 4]


def _competitor_bleed(answer_lower: str, context_lower: str) -> list[str]:
    """
    Return competitor entity names that appear in the answer but NOT in the
    retrieved context.  These are likely hallucinated comparisons or
    incorrectly attributed product behavior.
    """
    return [
        entity
        for entity in _COMPETITOR_ENTITIES
        if entity in answer_lower and entity not in context_lower
    ]


def evaluate(
    answer: str,
    retrieved_chunks: list[RetrievedChunk],
    query: str,
    chunk_embeddings: Optional[NDArray[np.float32]] = None,
) -> EvaluationResult:
    """
    Evaluate answer quality against retrieved evidence.

    Parameters
    ----------
    answer:
        The LLM-generated answer text.
    retrieved_chunks:
        Chunks used as context for the answer.
    query:
        Original (sanitized) user query.
    chunk_embeddings:
        Pre-computed normalized embeddings for retrieved_chunks, shaped
        (n_chunks, embed_dim).  If None, they will be computed here.
        Pass the cached value from RetrievalResult to avoid redundant work.
    """
    flags: list[str] = []
    answer_lower = answer.lower()

    is_refusal = any(p.search(answer) for p in REFUSAL_PATTERNS)
    if is_refusal:
        flags.append("REFUSAL_DETECTED")

    has_prompt_leakage = any(p.search(answer) for p in PROMPT_LEAKAGE_PATTERNS)
    if has_prompt_leakage:
        flags.append("PROMPT_LEAKAGE")

    has_domain_entities = any(entity in answer_lower for entity in CLEARPATH_ENTITIES)
    if not has_domain_entities:
        flags.append("NO_DOMAIN_ENTITIES")

    if not retrieved_chunks:
        flags.append("NO_CONTEXT")
        return EvaluationResult(
            confidence=Confidence.LOW,
            flags=flags,
            attribution_mean=0.0,
            attribution_min=0.0,
            attribution_variance=0.0,
            distinct_supporting_chunks=0,
            has_domain_entities=has_domain_entities,
            is_refusal=is_refusal,
            has_prompt_leakage=has_prompt_leakage,
        )

    # Domain guard: competitor bleed detection
    context_lower = " ".join(c.text for c in retrieved_chunks).lower()
    bled = _competitor_bleed(answer_lower, context_lower)
    if bled:
        flags.append(f"COMPETITOR_BLEED:{','.join(bled)}")

    # Sentence-level attribution
    sentences = _split_sentences(answer)[:8]
    attribution_scores: list[float] = []
    supporting_chunk_ids: set[str] = set()

    if sentences:
        model = _get_model()

        # Use pre-computed embeddings when available
        if chunk_embeddings is not None and chunk_embeddings.shape[0] == len(retrieved_chunks):
            c_embeddings = chunk_embeddings
        else:
            c_embeddings = model.encode(
                [c.text for c in retrieved_chunks],
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False,
            ).astype(np.float32)

        # Batch-encode all sentences at once for efficiency
        sent_embeddings: NDArray[np.float32] = model.encode(
            sentences,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        ).astype(np.float32)

        # (n_sentences, n_chunks)
        sim_matrix = sent_embeddings @ c_embeddings.T

        for sent_idx, row in enumerate(sim_matrix):
            max_sim = float(row.max())
            attribution_scores.append(max_sim)
            best_chunk_idx = int(row.argmax())
            if max_sim >= _ATTRIBUTION_SUPPORT_THRESHOLD:
                supporting_chunk_ids.add(retrieved_chunks[best_chunk_idx].chunk_id)

    if attribution_scores:
        attr_mean = float(np.mean(attribution_scores))
        attr_min = float(np.min(attribution_scores))
        attr_variance = float(np.var(attribution_scores))
    else:
        attr_mean = attr_min = attr_variance = 0.0

    distinct_supporting = len(supporting_chunk_ids)
    retrieved_count = len(retrieved_chunks)
    diversity_required = retrieved_count >= 3
    if diversity_required and distinct_supporting < MIN_DISTINCT_CHUNKS:
        flags.append("LOW_CITATION_DIVERSITY")

    # A HIGH confidence answer needs multi-chunk support only when enough chunks
    # were retrieved for that to be a reasonable expectation.
    diversity_satisfied = (not diversity_required) or (distinct_supporting >= MIN_DISTINCT_CHUNKS)

    # Confidence determination
    if has_prompt_leakage:
        confidence = Confidence.LOW
    elif is_refusal:
        confidence = Confidence.LOW
    elif bled:
        # Competitor bleed downgrades to at most MEDIUM
        if attr_mean >= 0.52 and diversity_satisfied:
            confidence = Confidence.MEDIUM
        else:
            confidence = Confidence.LOW
            if "LOW_ATTRIBUTION" not in flags:
                flags.append("LOW_ATTRIBUTION")
    elif attr_mean >= 0.52 and diversity_satisfied:
        confidence = Confidence.HIGH
    elif attr_mean >= 0.35:
        confidence = Confidence.MEDIUM
    else:
        confidence = Confidence.LOW
        flags.append("LOW_ATTRIBUTION")

    return EvaluationResult(
        confidence=confidence,
        flags=flags,
        attribution_mean=attr_mean,
        attribution_min=attr_min,
        attribution_variance=attr_variance,
        distinct_supporting_chunks=distinct_supporting,
        has_domain_entities=has_domain_entities,
        is_refusal=is_refusal,
        has_prompt_leakage=has_prompt_leakage,
    )