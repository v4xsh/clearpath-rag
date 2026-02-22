# written_answers.md

## Q1 — Routing Logic

The router has four conditions, and SIMPLE requires all of them to be true at once: word count at or under 12, exactly one question mark, no reasoning keywords, and no complaint tone. If any condition fails, the query is classified as COMPLEX.

I chose this boundary because it fails safely. Sending a simple query to the 70B model costs slightly more but still produces correct answers. Sending a complex query to the 8B model is where quality actually degrades — responses tend to be shallow or miss steps in multi-part troubleshooting. Because of that asymmetry, the router is intentionally biased toward over-routing to COMPLEX rather than under-routing.

Complaint detection is implemented with regex patterns rather than a keyword list. During testing, phrases like “is not triggering” and “isn’t firing” were consistently missed by simple substring matching. A static keyword set does not reliably capture multi-word negation patterns, while regex allows flexible spacing and phrasing.

One real misclassification observed during testing was:

> “What are the SSO configuration options?”

This query is short, contains one question mark, and has no reasoning or complaint signals, so it routes SIMPLE. However, the user is implicitly asking for a structured enumeration, and the 8B model sometimes returns an incomplete list.

To improve this without using an LLM, I would add a lightweight enumeration signal — for example detecting patterns like “what are” or “list all” followed by a plural noun. It would not need to be perfect, just catch the most obvious enumeration cases.

---

## Q2 — Retrieval Failures

Query:

> “Why does my scheduled report keep skipping the first row of data?”

This query is challenging because the user describes a symptom rather than using Clearpath terminology. The documentation does not use the same phrasing — it refers to this behavior as **offset configuration in pipeline export templates**.

In testing, retrieval returned three chunks about general pipeline scheduling, none of which mentioned data offsets or row-level export behavior. Coverage landed in the MEDIUM band, so the model generated an answer, but the result was a generic scheduling suggestion that did not resolve the issue.

The root cause is a mismatch between user language and documentation structure. BM25 contributes little because the query terms do not overlap with the relevant section. Semantic search ranks scheduling-related chunks highly because “scheduled report” is embedding-similar to pipeline scheduling — but that is the wrong subsystem.

The fix I would implement is lightweight query rewriting before retrieval. A deterministic synonym expansion layer could map common symptom phrases to technical terms (for example, “skipping rows” → “offset”, “row limit”, “export config”). This does not require an LLM — a curated lookup table would handle many high-frequency cases.

One additional failure mode I observed during testing is integration hallucination — the model sometimes confidently assumes Clearpath integrates with well-known tools (for example Jira) even when the retrieved context does not support it. This is a common grounding weakness in support RAG systems.

To mitigate this, I implemented a COMPETITOR_BLEED evaluator flag that detects when competitor product names appear in the answer but not in the retrieved context. When triggered, the response confidence is downgraded. This does not fully eliminate the behavior, but it provides an explicit safety signal and surfaces the risk in the debug panel.

---

## Q3 — Cost and Scale

Assume 5,000 queries per day with a 65% / 35% split between simple and complex queries.

**Simple queries (llama-3.1-8b-instant)**

- 3,250 × 900 input tokens = 2,925,000 input tokens  
- 3,250 × 120 output tokens = 390,000 output tokens  

Using approximate Groq proportional pricing:

- Input cost ≈ $0.146  
- Output cost ≈ $0.031  
- **Simple subtotal ≈ $0.18/day**

**Complex queries (llama-3.3-70b-versatile)**

- 1,750 × 2,200 input tokens = 3,850,000 input tokens  
- 1,750 × 220 output tokens = 385,000 output tokens  

- Input cost ≈ $2.27  
- Output cost ≈ $0.30  
- **Complex subtotal ≈ $2.57/day**

**Estimated total ≈ $2.75/day (~$82/month)**

The 70B model dominates cost despite handling only 35% of traffic. The main driver is large input context — complex queries retrieve up to eight chunks, significantly increasing prompt size.

The highest-ROI optimization is embedding-based response caching. Support traffic tends to repeat (password resets, user invites, etc.), and caching high-confidence responses would eliminate many LLM calls without architectural changes.

An optimization I would avoid is lowering the routing threshold to push more traffic to the 8B model. The potential savings are modest, while the quality drop on complaint and multi-step queries would be noticeable and user-visible.

---

## Q4 — What Is Broken

The attribution evaluator is somewhat brittle when answers are long but supporting chunks are short.

The evaluator embeds each answer sentence and computes cosine similarity against retrieved chunks. Sentences above a threshold are treated as supported. In practice, short chunks — such as brief API notes — can occasionally match multiple answer sentences because their embeddings are less semantically specific. This can inflate the attribution mean and make some answers appear more grounded than they truly are.

I shipped with this limitation because the evaluator still reliably catches higher-risk failures, including competitor hallucinations, missing domain entities, and prompt leakage. Slight attribution inflation in edge cases has not been the primary failure mode, and the sufficiency gate already blocks clearly weak retrieval scenarios.

The most direct fix would be to down-weight very short chunks in attribution scoring. For example, chunks under ~40 tokens could contribute less to the support calculation. In practice, these short segments are often headers or boundary fragments and should not carry the same evidential weight as full paragraphs.


## AI Usage

AI tools (primarily Claude and ChatGPT) were used during development as an engineering aid — mainly for debugging edge cases, validating implementation approaches, and tightening contract compliance. Core system design decisions (hybrid retrieval, deterministic routing, sufficiency gating, and evaluator logic) were implemented and tested manually.

Below is a representative example of the style of prompt used during later-stage hardening:

### Representative Prompt

> You are working on an existing production-quality FastAPI RAG system called Clearpath RAG.
> The system is already functionally correct.
> **Do not rewrite the architecture. Do not simplify the pipeline.**
>
> Your task is to make strict API contract alignment fixes and a few surgical production hardening improvements.
>
> Key constraints:
>
> * Preserve hybrid retrieval (FAISS + BM25)
> * Keep deterministic router behavior
> * Maintain conversational short-circuit
> * Do not break the frontend
>
> Focus areas:
>
> * API contract schema alignment
> * evaluator flag compliance
> * router safety guarantees
> * prompt-injection boundary verification
>
> This is a precision alignment task, not a redesign.

Similar targeted prompts were used when debugging retrieval edge cases, refining routing behavior, and validating evaluator signals. All AI-assisted suggestions were manually reviewed and verified against the running system before being incorporated.
