"""
Offline evaluation harness for the Clearpath RAG system.

Usage:
    python tests/eval_harness.py --base-url http://localhost:8000

Reports:
    - hit@k (relevant source retrieved)
    - grounding pass rate
    - routing distribution
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from dataclasses import dataclass, field
from typing import Optional

import httpx

BASE_URL = "http://localhost:8000"
API_ENDPOINT = "/api/v1/chat"
SESSION_PREFIX = "eval_"


@dataclass
class EvalCase:
    query: str
    expected_keywords: list[str]
    expected_routing: str  # SIMPLE or COMPLEX or CONVERSATIONAL
    description: str
    expected_behavior: str = "rag"  # "rag" | "conversational"


EVAL_CASES: list[EvalCase] = [
    EvalCase(
        query="How do I reset my password?",
        expected_keywords=["password", "reset"],
        expected_routing="SIMPLE",
        description="Simple password reset inquiry",
    ),
    EvalCase(
        query="What is Clearpath?",
        expected_keywords=["clearpath"],
        expected_routing="SIMPLE",
        description="Basic product identity question",
    ),
    EvalCase(
        query="How do I connect a data source?",
        expected_keywords=["data source", "connector", "integration"],
        expected_routing="SIMPLE",
        description="Data source connection inquiry",
    ),
    EvalCase(
        query="What pricing plans are available?",
        expected_keywords=["billing", "subscription", "plan"],
        expected_routing="SIMPLE",
        description="Pricing question",
    ),
    EvalCase(
        query="How do I invite team members to my workspace?",
        expected_keywords=["team", "workspace", "invite", "user"],
        expected_routing="SIMPLE",
        description="Team management question",
    ),
    EvalCase(
        query="Can you explain the difference between workspace roles and how each one affects what a user can do inside Clearpath?",
        expected_keywords=["user role", "access", "permission"],
        expected_routing="COMPLEX",
        description="Role comparison requiring explanation",
    ),
    EvalCase(
        query="Why isn't my webhook triggering after I set it up?",
        expected_keywords=["webhook"],
        expected_routing="COMPLEX",
        description="Complaint-toned debugging question",
    ),
    EvalCase(
        query="Compare the dashboard features available on the Professional plan versus the Enterprise plan.",
        expected_keywords=["dashboard", "plan"],
        expected_routing="COMPLEX",
        description="Cross-plan comparison",
    ),
    EvalCase(
        query="How do I enable SSO?",
        expected_keywords=["sso", "saml", "oauth"],
        expected_routing="SIMPLE",
        description="SSO configuration question",
    ),
    EvalCase(
        query="What export formats does Clearpath support?",
        expected_keywords=["export"],
        expected_routing="SIMPLE",
        description="Export formats inquiry",
    ),
    EvalCase(
        query="How does Clearpath handle audit logging and what events are captured in the audit trail?",
        expected_keywords=["audit log"],
        expected_routing="COMPLEX",
        description="Audit logging detailed explanation",
    ),
    EvalCase(
        query="Walk me through setting up an automated pipeline from scratch with a scheduled trigger.",
        expected_keywords=["pipeline", "automation", "schedule"],
        expected_routing="COMPLEX",
        description="Multi-step automation setup",
    ),
    EvalCase(
        query="What notifications can I configure?",
        expected_keywords=["notification", "alert"],
        expected_routing="SIMPLE",
        description="Notification configuration query",
    ),
    EvalCase(
        query="The integration is broken and data isn't syncing properly, what should I do?",
        expected_keywords=["integration"],
        expected_routing="COMPLEX",
        description="Integration failure complaint",
    ),
    EvalCase(
        query="How do API keys work in Clearpath?",
        expected_keywords=["api key"],
        expected_routing="SIMPLE",
        description="API key usage question",
    ),
    EvalCase(
        query="Explain how access control policies are evaluated when a user tries to view a report they don't own.",
        expected_keywords=["access control", "report", "user role"],
        expected_routing="COMPLEX",
        description="Access control policy explanation",
    ),
    EvalCase(
        query="Can I import data from a CSV file?",
        expected_keywords=["import"],
        expected_routing="SIMPLE",
        description="Data import question",
    ),
    EvalCase(
        query="What are the pros and cons of using OAuth versus SAML for single sign-on in Clearpath?",
        expected_keywords=["oauth", "saml", "sso"],
        expected_routing="COMPLEX",
        description="OAuth vs SAML comparison",
    ),
    EvalCase(
        query="How do I delete my organization account?",
        expected_keywords=["organization"],
        expected_routing="SIMPLE",
        description="Account deletion query",
    ),
    EvalCase(
        query="Why does Clearpath show an error when I try to schedule a report, and how do I fix it?",
        expected_keywords=["report", "schedule", "error"],
        expected_routing="COMPLEX",
        description="Scheduled report error debugging",
    ),
    EvalCase(
        query="hi",
        expected_keywords=[],
        expected_routing="CONVERSATIONAL",
        description="Pure greeting — must not enter RAG pipeline",
        expected_behavior="conversational",
    ),
    EvalCase(
        query="hello how are you",
        expected_keywords=[],
        expected_routing="CONVERSATIONAL",
        description="Greeting with small talk — must not enter RAG pipeline",
        expected_behavior="conversational",
    ),
]


@dataclass
class EvalResult:
    query: str
    description: str
    expected_routing: str
    actual_routing: str
    routing_correct: bool
    confidence: str
    grounded: bool
    hit_at_k: bool
    attribution_score: float
    flags: list[str]
    error: Optional[str] = None


@dataclass
class EvalReport:
    total: int
    errors: int
    hit_at_k: int
    grounding_passes: int
    routing_correct: int
    routing_distribution: dict[str, int] = field(default_factory=dict)
    mean_attribution: float = 0.0
    results: list[EvalResult] = field(default_factory=list)


async def run_case(
    client: httpx.AsyncClient, case: EvalCase, session_id: str
) -> EvalResult:
    try:
        response = await client.post(
            API_ENDPOINT,
            json={"query": case.query, "session_id": session_id},
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()

        answer_lower = data["answer"].lower()
        hit = any(kw.lower() in answer_lower for kw in case.expected_keywords)

        if case.expected_behavior == "conversational":
            actual_routing = "CONVERSATIONAL"
            routing_correct = data.get("model_used") == "none"
            # Conversational cases always pass hit@k if they short-circuited correctly
            hit = routing_correct
        else:
            actual_routing = "COMPLEX"
            if "SIMPLE" in data.get("routing_reason", "").upper() or data.get("model_used", "").endswith("8b-instant"):
                actual_routing = "SIMPLE"

            # Infer routing from model
            model = data.get("model_used", "")
            if "8b" in model:
                actual_routing = "SIMPLE"
            elif "70b" in model:
                actual_routing = "COMPLEX"

            routing_correct = actual_routing == case.expected_routing
        grounded = data["confidence"] in ("HIGH", "MEDIUM")

        return EvalResult(
            query=case.query,
            description=case.description,
            expected_routing=case.expected_routing,
            actual_routing=actual_routing,
            routing_correct=routing_correct,
            confidence=data["confidence"],
            grounded=grounded,
            hit_at_k=hit,
            attribution_score=data.get("attribution_score", 0.0),
            flags=data.get("flags", []),
        )
    except Exception as e:
        return EvalResult(
            query=case.query,
            description=case.description,
            expected_routing=case.expected_routing,
            actual_routing="UNKNOWN",
            routing_correct=False,
            confidence="LOW",
            grounded=False,
            hit_at_k=False,
            attribution_score=0.0,
            flags=[],
            error=str(e),
        )


async def run_eval(base_url: str) -> EvalReport:
    report = EvalReport(
        total=len(EVAL_CASES),
        errors=0,
        hit_at_k=0,
        grounding_passes=0,
        routing_correct=0,
        routing_distribution={"SIMPLE": 0, "COMPLEX": 0, "UNKNOWN": 0},
    )

    attribution_scores: list[float] = []

    async with httpx.AsyncClient(base_url=base_url) as client:
        for i, case in enumerate(EVAL_CASES):
            session_id = f"{SESSION_PREFIX}{i}"
            print(f"[{i+1:02d}/{len(EVAL_CASES)}] {case.description[:60]}")
            result = await run_case(client, case, session_id)
            report.results.append(result)

            if result.error:
                report.errors += 1
                print(f"         ERROR: {result.error}")
                continue

            if result.hit_at_k:
                report.hit_at_k += 1
            if result.grounded:
                report.grounding_passes += 1
            if result.routing_correct:
                report.routing_correct += 1

            routing = result.actual_routing
            report.routing_distribution[routing] = report.routing_distribution.get(routing, 0) + 1
            attribution_scores.append(result.attribution_score)

            status = "PASS" if result.hit_at_k else "FAIL"
            print(
                f"         {status} | route={result.actual_routing}({'ok' if result.routing_correct else 'WRONG'}) "
                f"| conf={result.confidence} | attr={result.attribution_score:.3f}"
            )

    if attribution_scores:
        report.mean_attribution = statistics.mean(attribution_scores)

    return report


def print_report(report: EvalReport) -> None:
    valid = report.total - report.errors
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Total cases:           {report.total}")
    print(f"Errors:                {report.errors}")
    print(f"Valid cases:           {valid}")
    print()
    if valid > 0:
        print(f"hit@k:                 {report.hit_at_k}/{valid} ({100*report.hit_at_k/valid:.1f}%)")
        print(f"Grounding pass rate:   {report.grounding_passes}/{valid} ({100*report.grounding_passes/valid:.1f}%)")
        print(f"Routing accuracy:      {report.routing_correct}/{valid} ({100*report.routing_correct/valid:.1f}%)")
        print(f"Mean attribution:      {report.mean_attribution:.4f}")
        print()
        print("Routing distribution:")
        for routing, count in sorted(report.routing_distribution.items()):
            print(f"  {routing}: {count}")
    print("=" * 60)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Clearpath RAG eval harness")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    args = parser.parse_args()

    print(f"Running {len(EVAL_CASES)} eval cases against {args.base_url}")
    report = await run_eval(args.base_url)
    print_report(report)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "total": report.total,
                    "errors": report.errors,
                    "hit_at_k": report.hit_at_k,
                    "grounding_passes": report.grounding_passes,
                    "routing_correct": report.routing_correct,
                    "mean_attribution": report.mean_attribution,
                    "routing_distribution": report.routing_distribution,
                    "results": [
                        {
                            "query": r.query,
                            "description": r.description,
                            "expected_routing": r.expected_routing,
                            "actual_routing": r.actual_routing,
                            "routing_correct": r.routing_correct,
                            "confidence": r.confidence,
                            "grounded": r.grounded,
                            "hit_at_k": r.hit_at_k,
                            "attribution_score": r.attribution_score,
                            "flags": r.flags,
                            "error": r.error,
                        }
                        for r in report.results
                    ],
                },
                f,
                indent=2,
            )
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())