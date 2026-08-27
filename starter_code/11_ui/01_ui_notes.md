# Phase 13b — Streamlit Stakeholder Demo

## Design

A two-tab Streamlit interface was built, calling the Phase 13a FastAPI service over HTTP rather than importing model code directly, mirroring how a real frontend would interact with a deployed backend, and testing the API contract itself as part of the demo.

- **Escalation Risk Predictor tab:** a form covering the classifier's key input fields (issue category, priority, region, channel, order value, delivery delay, previous tickets, tenure), returning a probability metric plus a clear color-coded status (green "low risk" vs. amber/red "flagged for human review"), directly surfacing the same 60% threshold used in the API.
- 
**Policy Assistant tab:** a free-text question box calling the RAG endpoint, displaying the answer alongside its cited sources and top retrieval score, giving a stakeholder visibility into why the assistant gave a particular answer, not just the answer itself.

## Error Handling

Both tabs wrap their API calls in try/except blocks catching requests.exceptions.RequestException, displaying a clear Streamlit error message ("Could not reach the API. Is it running?") rather than an unhandled crash if the FastAPI service isn't running or is unreachable.

## Live Demonstration Results
- **Escalation predictor:** a low-priority, London-region, web-chat ticket with no delivery delay returned 50.8% escalation probability, correctly displaying the "low risk" success state, consistent with Phase 4's finding that priority and issue category are strong escalation predictors (this case used neither of the highest-risk values).
- **Policy assistant:** "How many days do I have to return an item?" returned the same correct, multi-source, cited answer (30 days standard/electronics, 21 days marketplace) demonstrated throughout Phases 10, 12, and 13a, confirming consistent behavior across every interface exposing the RAG pipeline.

## Stakeholder Value

This demo gives a non-technical stakeholder (e.g. a support operations manager) a concrete, interactive way to see both the escalation model's risk scoring and the policy assistant's grounded, source-cited answers, without needing to interact with code, notebooks, or raw API calls directly.