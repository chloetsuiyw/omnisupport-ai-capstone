# Phase 11 — Agent & Tool Calling

## Architecture

Five local tools were implemented against the supplied agent_store CSVs and Phase 10's RAG pipeline: lookup_order, lookup_customer, check_return_eligibility, calculate_refund, search_policy. All CSV data is cached in memory on first use (module-level dict caching) rather than re-read on every call, per the starter code's explicit instruction not to repeatedly scan the local store.

The agent loop uses an LLM (google/gemini-2.5-flash) purely for intent routing, extracting structured intent, order ID, requested amount, and condition from free-text requests — while all actual business logic (approval thresholds, tool selection, missing-information checks) is enforced in code, not left to the LLM's discretion. This "LLM proposes, code disposes" design ensures safety-critical decisions (e.g. the £100 frontline refund limit) cannot be silently overridden by model behavior.

## Approval Logic

A £100 frontline authority limit was implemented for refunds. Critically, approval is triggered by whichever is higher: the tool's calculated refund, or the customer's explicitly requested amount, not the calculated amount alone. This closes a potential safety gap: a customer demanding an inflated amount (test case A03: customer demands £240, but the system's own calculation proposes only £100) still correctly triggers human approval, since routing on the calculated amount alone would have let a £240 demand through unchecked.

## Test Results

All 8 supplied test cases (evaluation/agent_test_cases.json) passed, matching both expected tool usage and expected human-approval routing:

| ID | Scenario | Tools Called (match?) | Approval Routing (match?) |
|---|---|---|---|
| A01 | Check order status | ✓ | ✓ |
| A02 | Calculate refund (within limit) | ✓ | ✓ |
| A03 | Refund demand exceeding limit | ✓ | ✓ |
| A04 | Missing order ID | ✓ (no tools called) | ✓ |
| A05 | Policy question | ✓ | ✓ |
| A06 | Account/identity change | ✓ (refused) | ✓ |
| A07 | Cancel dispatched order | ✓ | ✓ |
| A08 | Safety issue | ✓ | ✓ |

## Notable Behaviors
- **A04 (missing information):** the agent correctly asked for the order ID rather than guessing or calling a tool with an invalid/absent identifier, directly satisfying the brief's requirement to test "cases where the correct behaviour is to request missing information rather than call a tool."
- **A06 (account change):** the agent refused to attempt any account modification, correctly identifying this as requiring human identity verification, no tool exists (or should exist) for this action, and the agent did not attempt to fabricate one.
- **A07 (cancel dispatched order):** the agent inspected the order via lookup_order, correctly avoided claiming the dispatched order had been cancelled, and routed to human approval, demonstrating it does not perform or falsely confirm an action outside its safe authority.
- **A08 (safety issue):** the agent retrieved relevant safety policy via search_policy and escalated to a human, rather than either ignoring the safety signal or attempting to resolve it autonomously.

## Design Limitation

Intent routing relies on the LLM correctly extracting structured fields (intent, order ID, amount) from free text. While all 8 test cases succeeded, a genuinely malformed or adversarial request (e.g. an order ID embedded in unusual formatting, or a deliberately ambiguous multi-intent message) could cause misrouting. This is a natural candidate for the regression-testing and guardrails work in Phase 12, the 8 test cases here function as an initial regression suite, but a production system would benefit from a larger, more adversarial test set.