# Phase 9 — Prompting, Structured Outputs, and LLM Benchmarking

## Setup and Provider Trade-off (Free vs. Paid)

This phase used OpenRouter as the LLM API gateway, per course guidance. Initial testing used a free-tier model (google/gemma-4-26b-a4b-it:free) to avoid unnecessary cost. This model performed well on individual test calls but was unreliable under any sustained load, returning HTTP 429 errors from an upstream_provider_shared_pool, meaning free-tier capacity is shared across all users of that model globally, not allocated per-account. Even with exponential backoff retry logic (up to 5 attempts, delays up to 24s), the free tier could not reliably complete a batch of 5 sequential regression-test calls.

Decision: switched to a small paid top-up and google/gemini-2.5-flash, a low-cost, capable model. This resolved reliability issues immediately, all subsequent calls (regression tests, 20-request benchmark) completed without a single retry needed. This is a genuine, real-world trade-off worth documenting explicitly: free-tier LLM access is viable for occasional, low-volume testing, but not for any workflow requiring reliable throughput, a relevant consideration for anyone building a production system on a cost-constrained LLM budget. Given the eventual measured cost (see below), the paid tier's reliability was overwhelmingly worth the marginal cost for this project's needs.

## Structured Output Extraction

A Pydantic schema (TicketExtraction) defines the required output shape: issue category, priority, order ID, customer intent, a human-review flag, and any missing information. The extraction prompt instructs the LLM to return only a JSON object, which is then parsed and validated against the schema.

Example (real API output): given a message about a damaged item with an explicit order ID and frustrated tone, the model correctly extracted issue_category='damaged_item', priority='urgent', the order ID, a concise intent summary, and correctly flagged needs_human_review=True given the emotional tone, output that validated cleanly against the Pydantic schema with no errors.

## Prompt Regression Tests

A small suite of 5 hand-written test cases (with known expected categories, deliberately covering distinct categories) was run against the classification prompt to catch prompt drift or regressions from future prompt edits:

| Test Message | Expected | Result |
|---|---|---|
| "My package never arrived..." | lost_parcel | PASS |
| "The screen on my new phone is cracked" | damaged_item | PASS |
| "I can't log into my account..." | account_access | PASS |
| "When will my order arrive?..." | delivery_late | PASS |
| "I want to return this..." | wrong_item | PASS |

5/5 tests passed on the first run using the paid model, with zero retries needed.

## LLM Benchmark (20 Requests)

20 classification requests were run sequentially against google/gemini-2.5-flash, recording latency and token usage per request.

| Metric | Value |
|---|---|
| Mean latency | 0.541 sec |
| Latency range | 0.328s – 0.890s |
| Total tokens used | 1,568 |
| Mean tokens per request | 78.4 |
| Total input tokens | 1,512 |
| Total output tokens | 56 |
| Estimated total cost | $0.000594 |

## Cost and Latency Discussion

At $0.30/M input tokens and $2.50/M output tokens (OpenRouter's listed rate for Gemini 2.5 Flash), this 20-request benchmark cost under $0.0006 total, for a real business deployment processing, for example, 10,000 tickets/day through this classification prompt, that scales to roughly $0.30/day, a genuinely negligible operating cost relative to the value of automated triage.

Latency (mean 0.541s) is well within acceptable bounds for an asynchronous ticket-triage pipeline (not a live chat requiring sub-100ms response), though it would be a meaningful constraint for a real-time conversational use case. The latency variance observed (0.33s-0.89s, nearly 3x range) is worth noting as a reliability consideration: a production system should not assume consistent response times and should implement timeouts and fallback handling accordingly, a natural bridge into the guardrails and evaluation work in Phase 12.

## Trade-off Summary

This phase's central finding is not about any single prompt's accuracy (which was strong, 5/5 on regression tests) but about the infrastructure choice underlying LLM integration: free-tier access introduces real reliability risk that a low, predictable paid cost effectively eliminates. For a business-critical support triage pipeline, this cost-reliability trade-off would favor paid infrastructure without hesitation, given the cost is negligible relative to the operational risk of unreliable free-tier throughput.