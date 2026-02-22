"""
Deterministic rule-based query router.

SIMPLE classification requires ALL conditions to be true:
  - word_count <= 12
  - single question (question_count == 1)
  - no reasoning keywords
  - no complaint markers

Any conflicting signal defaults to COMPLEX.
Conflict resolution: COMPLEX always wins — there is no tie.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from app.config import settings
from app.rag.query_analyzer import QueryFeatures

__all__ = ["Classification", "RoutingDecision", "route"]


class Classification(str, Enum):
    SIMPLE = "SIMPLE"
    COMPLEX = "COMPLEX"


@dataclass
class RoutingDecision:
    classification: Classification
    model_used: str
    complexity_score: float
    triggered_rules: list[str]
    reasoning_trace: str


def route(features: QueryFeatures) -> RoutingDecision:
    triggered_rules: list[str] = []
    complex_signals: list[str] = []

    if features.word_count > settings.simple_max_words:
        complex_signals.append(f"word_count={features.word_count} > {settings.simple_max_words}")

    if features.question_count != 1:
        complex_signals.append(f"question_count={features.question_count} (expected 1)")

    if features.has_reasoning_keywords:
        complex_signals.append(
            f"reasoning_keywords={features.matched_reasoning_keywords}"
        )

    if features.has_complaint_tone:
        complex_signals.append(
            f"complaint_markers={features.matched_complaint_markers}"
        )

    is_simple = len(complex_signals) == 0

    if is_simple:
        classification = Classification.SIMPLE
        model_used = settings.model_simple
        triggered_rules = ["word_count_ok", "single_question", "no_reasoning", "no_complaint"]
        reasoning_trace = (
            f"All SIMPLE conditions satisfied: "
            f"words={features.word_count}, questions={features.question_count}, "
            f"no reasoning/complaint signals detected."
        )
    else:
        classification = Classification.COMPLEX
        model_used = settings.model_complex
        triggered_rules = complex_signals
        reasoning_trace = (
            f"COMPLEX due to: {'; '.join(complex_signals)}. "
            f"Conflict resolution: any non-simple signal promotes to COMPLEX."
        )

    decision = RoutingDecision(
        classification=classification,
        model_used=model_used,
        complexity_score=features.complexity_score,
        triggered_rules=triggered_rules,
        reasoning_trace=reasoning_trace,
    )
    assert decision.classification in (Classification.SIMPLE, Classification.COMPLEX), (
        f"Router returned unexpected classification: {decision.classification!r}"
    )
    return decision