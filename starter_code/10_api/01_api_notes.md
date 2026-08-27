# Phase 13a — FastAPI Service

## Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| /health | GET | Confirms the service is running and the model is loaded |
| /predict/escalation | POST | Returns escalation probability for a ticket, using the Phase 4 Random Forest classifier |
| /ask/policy | POST | Returns a grounded, cited policy answer, using the Phase 10 RAG pipeline |

All three endpoints were manually tested via the `/docs` interface and confirmed working: `/health` returned `{"status": "ok", "model_loaded": true}`, and both predictive endpoints returned correct, well-formed responses (detailed below).

## Design Decisions
- **Startup-time model loading:** the classifier and RAG index are both loaded once at server startup (@app.on_event("startup")), not on every request, avoiding the cost of retraining/re-embedding per call. This does mean the server takes roughly a minute to become ready after starting, a reasonable and standard trade-off for a service handling many subsequent fast requests.
- **Pydantic request/response schemas:** both predictive endpoints use explicit Pydantic models for input validation and output shape, so malformed requests are rejected with clear validation errors before reaching model code, and responses are guaranteed to match a documented, predictable schema.
- **Controlled errors:** /ask/policy returns an explicit 400 error for an empty question rather than passing it through to the RAG pipeline; /predict/escalation returns a 503 if the model isn't yet loaded, rather than crashing with an unhandled exception.

## Live Test Results

/ask/policy, question: "How many days do I have to return an item?" — correctly returned a multi-source, cited answer (30 days standard/electronics, 21 days marketplace), matching the same result quality demonstrated throughout Phase 10 and 12.

/predict/escalation, a lost_parcel + high priority ticket — returned escalation_probability: 0.7328, correctly flagging both predicted_escalation: true and requires_human_review: true. This result is consistent with Phase 4's finding that lost_parcel carries the highest escalation rate of any category in the dataset, confirming the API correctly serves the underlying model's learned behavior.

## Interactive Documentation

FastAPI's auto-generated /docs endpoint (Swagger UI) was used for manual testing, providing a browsable interface for both endpoints without needing a separate client tool, useful both for development testing and as a lightweight demonstration surface for stakeholders.