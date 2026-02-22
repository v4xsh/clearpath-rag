"""
Lightweight query feature extractor for routing and retrieval decisions.

keyword_density is defined as the fraction of query words that are
recognizable domain or technical terms.  This is used to shift hybrid
retrieval fusion weights toward BM25 when the query is term-heavy.
Typical values: 0.0–0.5 for natural language queries, higher for
technical/exact-term queries like "webhook HMAC signature error".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import tiktoken

_TOKENIZER = tiktoken.get_encoding("cl100k_base")

REASONING_KEYWORDS = frozenset(
    [
        "explain",
        "compare",
        "difference",
        "why",
        "how does",
        "analyze",
        "analyse",
        "elaborate",
        "describe in detail",
        "walk me through",
        "what causes",
        "tradeoff",
        "pros and cons",
        "versus",
        "evaluate",
    ]
)

# Complaint detection uses compiled regexes rather than plain substring matching
# so that multi-word negation patterns ("is not triggering", "isn't firing") are
# captured regardless of intervening whitespace variations.
_COMPLAINT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"not\s+working",
        r"not\s+triggering",
        r"not\s+firing",
        r"not\s+sending",
        r"not\s+receiving",
        r"not\s+loading",
        r"not\s+syncing",
        r"not\s+connecting",
        r"isn'?t\s+working",
        r"isn'?t\s+triggering",
        r"isn'?t\s+firing",
        r"doesn'?t\s+work",
        r"does\s+not\s+work",
        r"won'?t\s+work",
        r"won'?t\s+trigger",
        r"stopped\s+working",
        r"is\s+not\s+working",
        r"is\s+not\s+triggering",
        r"is\s+not\s+firing",
        r"\bbroken\b",
        r"\bbug\b",
        r"\bissue\b",
        r"\berror\b",
        r"\bfailed\b",
        r"\bfailing\b",
        r"\bcrash\b",
        r"\bcrashing\b",
        r"\bfrustr",
        r"\bterrible\b",
        r"\bawful\b",
        r"\bhorrible\b",
        r"\bworst\b",
        r"\buseless\b",
        r"fix\s+this",
    ]
]

AMBIGUITY_MARKERS = frozenset(
    [
        "it",
        "this",
        "that",
        "they",
        "thing",
        "stuff",
        "something",
        "somehow",
        "somewhere",
    ]
)

# Technical/domain terms that signal BM25 should be weighted higher.
# These are terms a user might type verbatim expecting exact-match retrieval.
_DOMAIN_TERMS: frozenset[str] = frozenset(
    [
        "api",
        "webhook",
        "saml",
        "oauth",
        "sso",
        "hmac",
        "csv",
        "json",
        "yaml",
        "cron",
        "uuid",
        "sdk",
        "ssl",
        "tls",
        "http",
        "https",
        "rest",
        "jwt",
        "token",
        "endpoint",
        "payload",
        "schema",
        "migration",
        "connector",
        "pipeline",
        "ingestion",
        "clearpath",
        "dashboard",
        "workspace",
        "integration",
        "s3",
        "bigquery",
        "postgresql",
        "mysql",
        "salesforce",
        "hubspot",
        "slack",
        "okta",
        "ldap",
        "rbac",
        "siem",
        "cdn",
        "vpc",
        "ip",
        "dns",
        "ttl",
        "cidr",
    ]
)

_QUESTION_PATTERN = re.compile(r"\?")
_WORD_PATTERN = re.compile(r"\b\w+\b")

# Matches only pure conversational messages — no trailing content allowed.
# Anchored with ^ and $ so "hi reset password" cannot match.
_CONVERSATIONAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^\s*(hi|hello|hey)[!.]?\s*$",
        r"^\s*how are you[?!.]?\s*$",
        r"^\s*good (morning|afternoon|evening)[!.]?\s*$",
        r"^\s*what'?s up[?!.]?\s*$",
    ]
]

# English stop words that carry no domain signal
_STOP_WORDS: frozenset[str] = frozenset(
    [
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "on", "at", "by", "for", "with", "about",
        "against", "between", "into", "through", "during", "before", "after",
        "above", "below", "from", "up", "down", "out", "off", "over", "under",
        "again", "further", "then", "once", "and", "or", "but", "if", "while",
        "i", "me", "my", "myself", "we", "our", "you", "your", "he", "she",
        "it", "they", "what", "which", "who", "whom", "this", "that", "these",
        "those", "am", "not", "no", "nor", "so", "yet", "both", "either",
        "neither", "each", "few", "more", "most", "other", "some", "such",
        "than", "too", "very", "just", "how", "when", "where", "why", "get",
        "set", "make", "use", "go", "also", "any", "all",
    ]
)


@dataclass
class QueryFeatures:
    token_length: int
    word_count: int
    question_count: int
    has_reasoning_keywords: bool
    has_complaint_tone: bool
    has_ambiguity: bool
    keyword_density: float
    is_conversational: bool = False
    matched_reasoning_keywords: list[str] = field(default_factory=list)
    matched_complaint_markers: list[str] = field(default_factory=list)
    complexity_score: float = 0.0


def _compute_keyword_density(words: list[str]) -> float:
    """
    Fraction of non-stop-word query tokens that are recognizable domain/tech terms.

    Range: 0.0 (no technical terms) → 1.0 (every content word is a domain term).
    Typical thresholds for fusion weight shift: >= 0.20.

    Examples:
        "how do I reset my password" → ~0.0 (no domain terms)
        "webhook HMAC signature verification" → ~0.75 (3/4 content words are domain terms)
        "saml oauth sso configuration" → 1.0
    """
    content_words = [w for w in words if w not in _STOP_WORDS and len(w) > 1]
    if not content_words:
        return 0.0
    domain_hits = sum(1 for w in content_words if w in _DOMAIN_TERMS)
    return domain_hits / len(content_words)


def analyze(query: str) -> QueryFeatures:
    lower = query.lower()
    words = _WORD_PATTERN.findall(lower)
    word_count = len(words)
    token_length = len(_TOKENIZER.encode(query))
    question_count = len(_QUESTION_PATTERN.findall(query))

    is_conversational = any(p.search(lower) for p in _CONVERSATIONAL_PATTERNS)

    matched_reasoning = [kw for kw in REASONING_KEYWORDS if kw in lower]
    matched_complaint = [
        m.group(0) for p in _COMPLAINT_PATTERNS if (m := p.search(lower))
    ]
    matched_ambiguity = [m for m in AMBIGUITY_MARKERS if m in words]

    has_reasoning_keywords = len(matched_reasoning) > 0
    has_complaint_tone = len(matched_complaint) > 0
    has_ambiguity = len(matched_ambiguity) > 0

    keyword_density = _compute_keyword_density(words)

    # complexity_score is logged only — does not control routing
    score = 0.0
    if word_count > 12:
        score += 0.25
    if question_count > 1:
        score += 0.15
    if has_reasoning_keywords:
        score += 0.30
    if has_complaint_tone:
        score += 0.15
    if has_ambiguity:
        score += 0.15
    complexity_score = min(score, 1.0)

    return QueryFeatures(
        token_length=token_length,
        word_count=word_count,
        question_count=question_count,
        has_reasoning_keywords=has_reasoning_keywords,
        has_complaint_tone=has_complaint_tone,
        has_ambiguity=has_ambiguity,
        keyword_density=keyword_density,
        is_conversational=is_conversational,
        matched_reasoning_keywords=matched_reasoning,
        matched_complaint_markers=matched_complaint,
        complexity_score=complexity_score,
    )