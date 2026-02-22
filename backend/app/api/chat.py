"""
POST /chat endpoint with full telemetry response.
POST /query contract-compliant endpoint (alias with reshaped response).
"""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from app.evaluator.evaluator import Confidence, evaluate
from app.memory.memory import ConversationMemory, Role
from app.rag.injection_defense import (
    SYSTEM_PROMPT_PREFIX,
    sanitize_query,
    wrap_context,
)
from app.rag.llm_client import call_with_metrics
from app.rag.query_analyzer import analyze
from app.rag.retriever import RetrievedChunk, retriever
from app.rag.sufficiency import (
    FALLBACK_MESSAGE,
    classify_sufficiency,
    expanded_top_k,
    should_generate,
)
from app.router.router import Classification, route

logger = logging.getLogger(__name__)
router = APIRouter()

_sessions: dict[str, ConversationMemory] = {}


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    session_id: str = Field(default="default")


class SourceReference(BaseModel):
    chunk_id: str
    doc_id: str
    section_path: str
    page_number: int


class ChatResponse(BaseModel):
    answer: str
    confidence: str
    model_used: str
    tokens_input: int
    tokens_output: int
    latency_ms: float
    routing_reason: str
    complexity_score: float
    retrieval_k: int
    coverage_score: float
    attribution_score: float
    flags: list[str]
    sources: list[SourceReference]


# ---------------------------------------------------------------------------
# Contract-compliant /query schema
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """
    Accepts ``question`` (contract primary field) or ``query`` (backward-compat).
    ``question`` takes precedence when both are supplied.
    A 422 validation error is raised when neither field is present.
    """

    question: str | None = Field(default=None, min_length=1, max_length=4096)
    query: str | None = Field(default=None, min_length=1, max_length=4096)
    conversation_id: str = Field(default="")

    @model_validator(mode="after")
    def _require_one_field(self) -> "QueryRequest":
        if not (self.question or self.query):
            raise ValueError("Either 'question' or 'query' must be provided.")
        return self

    @property
    def resolved_question(self) -> str:
        """Return the normalised question text, preferring ``question``."""
        return (self.question or self.query or "").strip()


class TokenUsage(BaseModel):
    input: int
    output: int


class QueryMetadata(BaseModel):
    model_used: str
    classification: str          # "simple" | "complex"
    tokens: TokenUsage
    latency_ms: float
    chunks_retrieved: int
    evaluator_flags: list[str]


class QuerySource(BaseModel):
    document: str
    page: int
    # relevance_score is clamped to [0, 1] in _to_query_sources;
    # retriever scoring logic is not modified.
    relevance_score: float


class QueryResponse(BaseModel):
    answer: str
    metadata: QueryMetadata
    sources: list[QuerySource]
    conversation_id: str


# ---------------------------------------------------------------------------
# /query adapter utilities  (never called from /chat)
# ---------------------------------------------------------------------------

_FLAG_MAP: dict[str, str] = {
    "NO_CONTEXT": "no_context",
    "REFUSAL_DETECTED": "refusal",
    "PROMPT_LEAKAGE": "prompt_leakage",
    "LOW_CITATION_DIVERSITY": "low_citation_diversity",
    "LOW_ATTRIBUTION": "low_attribution",
    "NO_DOMAIN_ENTITIES": "no_domain_entities",
    "INSUFFICIENT_RETRIEVAL": "insufficient_retrieval",
}


def _normalize_flags(raw_flags: list[str]) -> list[str]:
    """
    Map internal uppercase evaluator flags to contract-compliant lowercase
    semantic names without modifying evaluator.py logic.

    - Known flags  → mapped via _FLAG_MAP.
    - COMPETITOR_BLEED:*  prefix → ``competitor_bleed``.
    - INJECTION_REMOVED:* prefix → ``injection_removed``.
    - Unknown flags → lowercased safely.
    """
    out: list[str] = []
    for flag in raw_flags:
        if flag.startswith("COMPETITOR_BLEED:"):
            out.append("competitor_bleed")
        elif flag.startswith("INJECTION_REMOVED:"):
            out.append("injection_removed")
        else:
            out.append(_FLAG_MAP.get(flag, flag.lower()))
    return out


def _ensure_no_context_flag(
    flags: list[str], chunks_retrieved: int, is_refusal: bool
) -> list[str]:
    """
    Contract requirement: ``no_context`` must be present whenever
    chunks_retrieved == 0 and the answer is not a refusal.
    Implemented in the adapter layer; evaluator logic is not modified.
    """
    if chunks_retrieved == 0 and not is_refusal and "no_context" not in flags:
        return ["no_context"] + flags
    return flags


def _classification_label(classification: Classification) -> str:
    """Map internal Classification enum to contract lowercase string."""
    return classification.value.lower()


def _to_query_sources(chunks: list[RetrievedChunk]) -> list[QuerySource]:
    return [
        QuerySource(
            document=c.doc_id,
            page=c.page_number,
            relevance_score=round(min(max(c.semantic_score, 0.0), 1.0), 4),
        )
        for c in chunks
    ]


# ---------------------------------------------------------------------------
# /chat endpoint — unchanged, frontend compatible
# ---------------------------------------------------------------------------


def _get_memory(session_id: str) -> ConversationMemory:
    if session_id not in _sessions:
        _sessions[session_id] = ConversationMemory()
    return _sessions[session_id]


def _build_system_prompt(chunks: list[RetrievedChunk]) -> str:
    context_parts = [
        f"[Source {i + 1}: {c.doc_id} / {c.section_path} (p.{c.page_number})]\n{c.text}"
        for i, c in enumerate(chunks)
    ]
    context_text = "\n\n---\n\n".join(context_parts)
    return SYSTEM_PROMPT_PREFIX + "\n\n" + wrap_context(context_text)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    start_time = time.monotonic()

    sanitized_query, removed_patterns = sanitize_query(request.query)
    injection_flags = [f"INJECTION_REMOVED:{p}" for p in removed_patterns]

    features = analyze(sanitized_query)

    if features.is_conversational:
        latency_ms = (time.monotonic() - start_time) * 1000
        lower = sanitized_query.lower()
        if "how are you" in lower:
            answer = "I'm doing great and ready to help with Clearpath questions. What can I assist you with?"
        else:
            answer = "Hi! I'm the Clearpath support assistant. How can I help you today?"

        memory = _get_memory(request.session_id)
        memory.add_turn(Role.USER, sanitized_query, source="user")
        memory.add_turn(Role.ASSISTANT, answer, source="conversational", confidence=Confidence.LOW)

        logger.info(
            "query=%r session=%s conversational latency_ms=%.1f",
            sanitized_query, request.session_id, latency_ms,
        )

        return ChatResponse(
            answer=answer,
            confidence=Confidence.LOW.value,
            model_used="none",
            tokens_input=0,
            tokens_output=0,
            latency_ms=latency_ms,
            routing_reason="conversational_short_circuit",
            complexity_score=0.0,
            retrieval_k=0,
            coverage_score=0.0,
            attribution_score=0.0,
            flags=[],
            sources=[],
        )

    routing = route(features)

    result = retriever.retrieve(
        query=sanitized_query,
        classification=routing.classification,
        keyword_density=features.keyword_density,
    )

    sufficiency_tier = classify_sufficiency(result.coverage_score)

    final_chunks = result.chunks
    final_k = result.top_k
    final_embeddings = result.chunk_embeddings

    if not should_generate(sufficiency_tier):
        answer = FALLBACK_MESSAGE
        eval_result = evaluate(answer, [], sanitized_query)
        latency_ms = (time.monotonic() - start_time) * 1000

        memory = _get_memory(request.session_id)
        memory.add_turn(Role.USER, sanitized_query, source="user")
        memory.add_turn(Role.ASSISTANT, answer, source="fallback", confidence=Confidence.LOW)

        logger.info(
            "query=%r session=%s tier=LOW model=%s latency_ms=%.1f",
            sanitized_query, request.session_id, routing.model_used, latency_ms,
        )

        return ChatResponse(
            answer=answer,
            confidence=Confidence.LOW.value,
            model_used=routing.model_used,
            tokens_input=0,
            tokens_output=0,
            latency_ms=latency_ms,
            routing_reason=routing.reasoning_trace,
            complexity_score=routing.complexity_score,
            retrieval_k=final_k,
            coverage_score=result.coverage_score,
            attribution_score=0.0,
            flags=["INSUFFICIENT_RETRIEVAL"] + injection_flags,
            sources=[],
        )

    # MEDIUM tier: expand retrieval slightly
    if sufficiency_tier.value == "MEDIUM":
        expanded_k = expanded_top_k(sufficiency_tier, result.top_k)
        if expanded_k > result.top_k:
            expanded_result = retriever.retrieve(
                query=sanitized_query,
                classification=routing.classification,
                keyword_density=features.keyword_density,
                top_k_override=expanded_k,
            )
            final_chunks = expanded_result.chunks
            final_k = expanded_k
            final_embeddings = expanded_result.chunk_embeddings

    system_prompt = _build_system_prompt(final_chunks)

    memory = _get_memory(request.session_id)
    memory.add_turn(Role.USER, sanitized_query, source="user")
    messages = memory.to_messages()

    try:
        llm_response = await call_with_metrics(
            system_prompt=system_prompt,
            messages=messages,
            model=routing.model_used,
        )
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        raise HTTPException(status_code=503, detail="LLM service unavailable")

    eval_result = evaluate(
        answer=llm_response.content,
        retrieved_chunks=final_chunks,
        query=sanitized_query,
        chunk_embeddings=final_embeddings if final_embeddings.size > 0 else None,
    )

    memory.add_turn(
        Role.ASSISTANT,
        llm_response.content,
        source=routing.model_used,
        confidence=eval_result.confidence,
    )

    latency_ms = (time.monotonic() - start_time) * 1000
    all_flags = eval_result.flags + injection_flags

    sources = [
        SourceReference(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            section_path=c.section_path,
            page_number=c.page_number,
        )
        for c in final_chunks
    ]

    logger.info(
        "query=%r session=%s tier=%s model=%s confidence=%s "
        "tokens_in=%d tokens_out=%d latency_ms=%.1f flags=%s",
        sanitized_query,
        request.session_id,
        sufficiency_tier.value,
        routing.model_used,
        eval_result.confidence.value,
        llm_response.tokens_input,
        llm_response.tokens_output,
        latency_ms,
        all_flags,
    )

    return ChatResponse(
        answer=llm_response.content,
        confidence=eval_result.confidence.value,
        model_used=routing.model_used,
        tokens_input=llm_response.tokens_input,
        tokens_output=llm_response.tokens_output,
        latency_ms=latency_ms,
        routing_reason=routing.reasoning_trace,
        complexity_score=routing.complexity_score,
        retrieval_k=final_k,
        coverage_score=result.coverage_score,
        attribution_score=eval_result.attribution_mean,
        flags=all_flags,
        sources=sources,
    )


@router.delete("/chat/session/{session_id}")
async def clear_session(session_id: str) -> dict:
    if session_id in _sessions:
        _sessions[session_id].clear()
    return {"cleared": True, "session_id": session_id}


# ---------------------------------------------------------------------------
# Contract-compliant /query endpoint
# ---------------------------------------------------------------------------


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Contract-compliant endpoint. Accepts ``question`` (primary) or ``query``
    (compat). Returns the reshaped API spec response with lowercase flags and
    nested metadata.
    """
    conversation_id = request.conversation_id or str(uuid.uuid4())
    normalized_question = request.resolved_question
    start_time = time.monotonic()

    sanitized_query, removed_patterns = sanitize_query(normalized_question)
    injection_flags = [f"INJECTION_REMOVED:{p}" for p in removed_patterns]

    features = analyze(sanitized_query)

    if features.is_conversational:
        latency_ms = (time.monotonic() - start_time) * 1000
        lower = sanitized_query.lower()
        if "how are you" in lower:
            answer = "I'm doing great and ready to help with Clearpath questions. What can I assist you with?"
        else:
            answer = "Hi! I'm the Clearpath support assistant. How can I help you today?"

        memory = _get_memory(conversation_id)
        memory.add_turn(Role.USER, sanitized_query, source="user")
        memory.add_turn(Role.ASSISTANT, answer, source="conversational", confidence=Confidence.LOW)

        # Fix 3: classification must be "simple"|"complex", never "none"
        return QueryResponse(
            answer=answer,
            metadata=QueryMetadata(
                model_used="none",
                classification="simple",
                tokens=TokenUsage(input=0, output=0),
                latency_ms=latency_ms,
                chunks_retrieved=0,
                evaluator_flags=[],
            ),
            sources=[],
            conversation_id=conversation_id,
        )

    routing = route(features)

    result = retriever.retrieve(
        query=sanitized_query,
        classification=routing.classification,
        keyword_density=features.keyword_density,
    )

    sufficiency_tier = classify_sufficiency(result.coverage_score)
    final_chunks = result.chunks
    final_embeddings = result.chunk_embeddings

    if not should_generate(sufficiency_tier):
        answer = FALLBACK_MESSAGE
        eval_result = evaluate(answer, [], sanitized_query)
        latency_ms = (time.monotonic() - start_time) * 1000

        memory = _get_memory(conversation_id)
        memory.add_turn(Role.USER, sanitized_query, source="user")
        memory.add_turn(Role.ASSISTANT, answer, source="fallback", confidence=Confidence.LOW)

        raw_flags = ["INSUFFICIENT_RETRIEVAL"] + eval_result.flags + injection_flags
        norm_flags = _normalize_flags(raw_flags)
        norm_flags = _ensure_no_context_flag(norm_flags, 0, eval_result.is_refusal)

        return QueryResponse(
            answer=answer,
            metadata=QueryMetadata(
                model_used=routing.model_used,
                classification=_classification_label(routing.classification),
                tokens=TokenUsage(input=0, output=0),
                latency_ms=latency_ms,
                chunks_retrieved=0,
                evaluator_flags=norm_flags,
            ),
            sources=[],
            conversation_id=conversation_id,
        )

    if sufficiency_tier.value == "MEDIUM":
        expanded_k = expanded_top_k(sufficiency_tier, result.top_k)
        if expanded_k > result.top_k:
            expanded_result = retriever.retrieve(
                query=sanitized_query,
                classification=routing.classification,
                keyword_density=features.keyword_density,
                top_k_override=expanded_k,
            )
            final_chunks = expanded_result.chunks
            final_embeddings = expanded_result.chunk_embeddings

    system_prompt = _build_system_prompt(final_chunks)

    memory = _get_memory(conversation_id)
    memory.add_turn(Role.USER, sanitized_query, source="user")
    messages = memory.to_messages()

    try:
        llm_response = await call_with_metrics(
            system_prompt=system_prompt,
            messages=messages,
            model=routing.model_used,
        )
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        raise HTTPException(status_code=503, detail="LLM service unavailable")

    eval_result = evaluate(
        answer=llm_response.content,
        retrieved_chunks=final_chunks,
        query=sanitized_query,
        chunk_embeddings=final_embeddings if final_embeddings.size > 0 else None,
    )

    memory.add_turn(
        Role.ASSISTANT,
        llm_response.content,
        source=routing.model_used,
        confidence=eval_result.confidence,
    )

    latency_ms = (time.monotonic() - start_time) * 1000
    raw_flags = eval_result.flags + injection_flags
    norm_flags = _normalize_flags(raw_flags)
    norm_flags = _ensure_no_context_flag(norm_flags, len(final_chunks), eval_result.is_refusal)

    logger.info(
        "question=%r conversation=%s tier=%s model=%s classification=%s "
        "tokens_in=%d tokens_out=%d latency_ms=%.1f flags=%s",
        sanitized_query,
        conversation_id,
        sufficiency_tier.value,
        routing.model_used,
        routing.classification.value,
        llm_response.tokens_input,
        llm_response.tokens_output,
        latency_ms,
        norm_flags,
    )

    return QueryResponse(
        answer=llm_response.content,
        metadata=QueryMetadata(
            model_used=routing.model_used,
            classification=_classification_label(routing.classification),
            tokens=TokenUsage(
                input=llm_response.tokens_input,
                output=llm_response.tokens_output,
            ),
            latency_ms=latency_ms,
            chunks_retrieved=len(final_chunks),
            evaluator_flags=norm_flags,
        ),
        sources=_to_query_sources(final_chunks),
        conversation_id=conversation_id,
    )