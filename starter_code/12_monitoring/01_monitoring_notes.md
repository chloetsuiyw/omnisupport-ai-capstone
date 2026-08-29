# Phase 14 — Docker, CI/CD, and Monitoring

## Objective
Containerize the OmniSupport AI API, establish a passing CI pipeline, and produce a monitoring summary covering traffic, latency, retrieval quality, and reliability.

## GitHub Repository & CI
Repo live at github.com/chloetsuiyw/omnisupport-ai-capstone. CI pipeline (tests.yml) via GitHub Actions, fixed after an initial failure caused by a missing pandas dependency in requirements-ci.txt. LLM_API_KEY and MODEL_NAME configured as GitHub repository secrets and injected into the test job's env block, since rag_pipeline.py instantiates the OpenAI client at module import time, without this, importing the module during test collection raises an error even when the key isn't otherwise needed for the specific test running.

## Dockerfile
CMD updated from the original stub target (app.api.main:app) to uvicorn main:app --app-dir starter_code/10_api. Root cause: 10_api is not a valid Python module name (identifiers can't start with a digit), so a dotted import path fails; --app-dir sidesteps this by changing uvicorn's working directory for module resolution instead of importing through the numbered package path. requirements.txt updated to include sentence-transformers, openai, accelerate, streamlit were added incrementally across earlier phases locally but not yet reflected in the file used for the Docker build.

## Docker Desktop Setup
Initial blocker: Docker Desktop failed to start due to virtualization not being fully enabled. 
Fix: enabled Intel VT-x/VT-d in BIOS, then completed Windows-side Hyper-V/WSL2 configuration. Once resolved, docker build completed successfully (all 5 layers, cached on subsequent builds).

## Container Run & Verification
Ran the built image locally with docker run -p 8000:8000 --env-file .env omnisupport-ai. Verified in order: clean startup logs (embedding model loaded, no tracebacks, Uvicorn running), /docs loading correctly with all three endpoints listed, /health returning 200, and finally /ask/policy, the real end-to-end test, which initially failed with 500 Internal Server Error.

Bug Found and Fixed: Malformed API Key from .env Whitespace Symptom: /ask/policy returned a generic 500. Server-side traceback showed openai.AuthenticationError: Error code: 401 - {'error': {'message': 'Missing Authentication header', 'code': 401}}.

Diagnosis: confirmed via docker exec ... env that both LLM_API_KEY and MODEL_NAME were present inside the container, ruling out --env-file not loading at all. Confirmed via code inspection that the client reads os.getenv("LLM_API_KEY"), matching the .env variable name, ruling out a naming mismatch. The specific wording ("Missing Authentication header," not a generic invalid-key message) indicated OpenRouter received an effectively empty/malformed Authorization header. A length/preview diagnostic run inside the container showed a double space between the preview: label and the value, indicating the key string carried a leading space character (length 74 vs. expected 73).

Root cause: .env had a space after the = sign (LLM_API_KEY= sk-or-...), which python-dotenv preserved literally as part of the value, producing a malformed Authorization: Bearer sk-or-... header that OpenRouter's parser rejected.

Fix: removed the stray space in .env. No rebuild required since .env is read at docker run time via --env-file, not baked into the image. Restarted the container; diagnostic confirmed 73 characters with no leading space. /ask/policy then returned 200 with a correctly grounded, cited answer.

## Monitoring Summary
Full metrics captured in monitoring/monitoring_summary_template.csv, generated via monitoring.py (Session 30 starter, completed with record_request, record_rag_outcome, and record_agent_tool_failure implementations), drawing on Phase 9 (LLM benchmark), Phase 10 (RAG retrieval behavior), Phase 12a (RAG evaluation), Phase 12b (prompt evaluation), and this phase's live container test.

Latency & cost (Phase 9, 20-request benchmark on google/gemini-2.5-flash): mean latency 541ms, range 328–890ms, acceptable for asynchronous ticket triage, not real-time chat. 1,568 total tokens across 20 requests, cost $0.000594, extrapolating to roughly $0.30/day at 10,000 tickets/day. Earlier free-tier testing hit persistent 429 rate limits even with exponential backoff; switching to paid tier resolved this immediately, a cost-reliability trade-off worth flagging explicitly.

Retrieval quality (Phase 10 + Phase 12a): average top retrieval score across a 15-question independent eval set was 0.547. 13/15 (87%) answered confidently and correctly with citations; 2/15 (13.3%) correctly refused rather than fabricated, with no hallucinated claims observed. Phase 10 identified a clear separation between answerable questions (score 0.50–0.70) and a deliberately out-of-scope question (score 0.17–0.18). This session's live container test scored 0.642 on a well-covered question but still produced hedged, partial-answer language, reinforcing that even "good" scores warrant caution.

Prompt safety (Phase 12b, 10 test cases): 10/10 schema-complete; 3/3 safety-critical cases handled correctly (refused unsafe secret collection, correctly distinguished a proposed £250 refund from a completed action, avoided an unsupported compatibility claim).

Final summary values: request_count 22, average_latency_ms 546.36, failure_count 1, rag_no_answer_rate 0.1333, agent_tool_failure_count 0, model_version google/gemini-2.5-flash, prompt_version rag_policy_prompt_v1.

Proposed human-review threshold: top_retrieval_score < 0.5. This sits just above the observed out-of-scope ceiling (0.18) and below the level (0.642) that still needed hedged language in live testing, giving it grounding in two independent pieces of evidence rather than being an arbitrary cutoff.

Note on data provenance: This summary was generated by backfilling monitoring.py with results from Phases 9, 10, and 12's evaluation runs plus one live post-fix container call, rather than from persistent live request logging. No request-level logging was implemented for this project's scope, consistent with the Auditability risk already flagged in 03_responsible_ai_risk_register.md ("traces are currently returned in-process but not persisted to a durable audit log"). The resulting numbers are a faithful snapshot of validated system behavior across the project, not a live production monitoring feed.

Data-Protection Note (Carried from ProviderConfig Finding)
ProviderConfig.from_environment() silently defaults to https://api.openai.com/v1 and gpt-4.1-mini when only LLM_API_KEY is set without an explicit base URL override. This wasn't the cause of the bug found here (this pipeline hardcodes the OpenRouter base URL directly), but it remains a documented control gap. Recommend surfacing the resolved base URL and model name in a startup log line so this class of misconfiguration is visible immediately rather than only at first failed request.

## Stakeholder Decision
Ship the container as verified. Before production rollout, implement the 0.5 retrieval-score human-review threshold as an actual routing rule, and consider adding persistent request-level logging so future monitoring summaries reflect live traffic rather than backfilled evaluation results.