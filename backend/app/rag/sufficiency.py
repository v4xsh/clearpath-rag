"""
3-tier retrieval sufficiency calibration.
"""

from __future__ import annotations

from enum import Enum

from app.config import settings


class SufficiencyTier(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


FALLBACK_MESSAGE = (
    "I'm here to help with Clearpath-related questions. I wasn't able to find "
    "relevant documentation for that query. Please check the official Clearpath "
    "documentation or contact support if you need further assistance."
)


def classify_sufficiency(coverage_score: float) -> SufficiencyTier:
    if coverage_score >= settings.coverage_high_threshold:
        return SufficiencyTier.HIGH
    if coverage_score >= settings.coverage_low_threshold:
        return SufficiencyTier.MEDIUM
    return SufficiencyTier.LOW


def should_generate(tier: SufficiencyTier) -> bool:
    return tier in (SufficiencyTier.HIGH, SufficiencyTier.MEDIUM)


def expanded_top_k(tier: SufficiencyTier, base_k: int) -> int:
    if tier == SufficiencyTier.MEDIUM:
        return base_k + 2
    return base_k