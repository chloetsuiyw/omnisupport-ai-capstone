# OmniSupport AI — Final Capstone Report

An integrated AI system for a retail support operation: predictive models for resolution time and escalation risk, a customer segmentation, a computer-vision return classifier, classical and semantic text search, a RAG policy assistant, a tool-using support agent, and a deployed API/UI, wrapped in evaluation, responsible-AI, and monitoring evidence.

This README is the project's architecture and evidence guide. Each section below states what was built, what was found, and links to the phase folder containing full detail, code, and evaluation artifacts. It is written to be read end-to-end for a fast overview, or used as a lookup table pointing to deeper evidence during review.

## System Architecture

```
                        ┌─────────────────────────┐
                        │   1,000,000-row ticket    │
                        │   dataset (10 Parquet     │
                        │   shards, data/raw/)      │
                        └───────────┬───────────────┘
                                    │
                cleaned, leakage-audited feature set
                                    │
        ┌───────────────┬──────────┼──────────┬──────────────────┐
        ▼               ▼          ▼          ▼                  ▼
  Resolution-time  Escalation  Customer   Tabular NN      (text/image
  regressor (RF)   classifier  clustering (comparison)     subsets below)
        │               │          │          │
        └───────┬───────┴──────────┴──────────┘
                ▼
     FastAPI service (/predict/escalation)
                │
                ▼
        Streamlit UI (stakeholder demo)


  Ticket text subset ──► TF-IDF classifier ──► compared against ──► Fine-tuned DistilBERT
                    └──► Semantic search (embeddings)

  480 return images ──► Basic CNN (from scratch) ──► compared against ──► ResNet18 transfer learning

  40 policy documents ──► chunked/embedded ──► RAG pipeline ──► FastAPI (/ask/policy) ──► Streamlit UI
                                                     │
                                        Agent (tool-calling, human-approval gates)
                                                     │
                              Docker container ──► CI (GitHub Actions) ──► monitoring.py
```

All predictive and generative components are served through one FastAPI service (`starter_code/10_api/main.py`) and one Streamlit UI (`starter_code/11_ui/app.py`), containerized via `deployment/Dockerfile`, tested via `.github/workflows/tests.yml`, and observed via `starter_code/12_monitoring/monitoring.py`.

## Setup and Run Instructions

**Environment:** Python 3.12, conda environment `omnisupport`.

```bash
pip install -r requirements.txt --break-system-packages   # or use conda/venv as preferred
```

**Environment variables** (`.env`, not committed — see `.env.example` if present):
```
LLM_API_KEY=<your OpenRouter API key>
MODEL_NAME=google/gemini-2.5-flash
```

**Run the API:**
```bash
python starter_code/10_api/main.py
# or: uvicorn main:app --app-dir starter_code/10_api
```
Visit `http://localhost:8000/docs` for interactive Swagger documentation.

**Run the UI** (with the API already running):
```bash
streamlit run starter_code/11_ui/app.py
```

**Run via Docker:**
```bash
docker build -f deployment/Dockerfile -t omnisupport-ai .
docker run -p 8000:8000 --env-file .env omnisupport-ai
```

**Run tests:**
```bash
pytest tests/
```

All scripts assume execution from the project root (relative paths such as `data/raw/...` and `knowledge_base/...` depend on this).

## Evidence Map (by rubric criterion)

### 1. Problem framing and understanding
`starter_code/01_data/01_problem_framing.md`

Two targets defined against a real operational pain point: resolution time (regression) and escalation (classification), both scored at ticket-creation time. Success metrics chosen deliberately for business interpretability (MAE over RMSE for staffing; PR-AUC/F1 plus a threshold table over accuracy for the imbalanced escalation target). A five-column leakage audit explicitly excludes post-outcome fields (`refund_amount`, `csat_score`, `customer_sentiment`, etc.) from the feature set, and `resolution_time_hours` is separately excluded from the escalation classifier's features as an outcome of the same underlying process rather than a valid predictor. A dedicated Risks and Assumptions section names five project-level risks (synthetic-data realism, moderate escalation-model recall, fixed business thresholds, limited subgroup-fairness scope, and static knowledge-base/agent-data freshness), each grounded in a specific finding from later phases rather than stated generically.

### 2. Data quality and preprocessing
`starter_code/01_data/02_data_audit_summary.md`

Full audit of 1,000,000 rows across 10 shards: two distinct missingness patterns (true NaN in `csat_score`; blank-string missingness in `customer_region`/`issue_description` that a plain `.isna()` check would miss), 4,000 exact duplicates removed, three casing inconsistencies merged, and column-specific outlier handling (capping with a retained flag for `order_value`; no action for `delivery_delay_days`, since IQR produces a meaningless negative bound on a floor-bounded count variable). A distinct cluster of 348 rows with resolution time > 300 hours was traced to a definitional relationship with `resolution_status_after_7d = "open_after_7d"` — this same anomaly resurfaces independently in the Phase 4 regression error analysis and the Phase 5 clustering validation (Cluster 3), which is used later as a positive validation signal.

### 3. EDA and feature engineering
Covered within `01_data/02_data_audit_summary.md` and `02_ml/01_regression_notes.md` / `02_classification_notes.md`.

Key EDA finding: escalation rate varies from ~24% to ~64% by `issue_category` (lost_parcel highest), directly motivating its use as a leading feature and later confirmed as the dominant feature by importance in both models. All engineered features (e.g. `order_value_was_capped`) are prediction-time valid per the Phase 1 leakage audit.

### 4. Classical ML
`starter_code/02_ml/01_regression_notes.md`, `02_classification_notes.md`

- **Regression:** Random Forest MAE 3.76 hrs vs. DummyRegressor baseline 6.77 hrs (44% reduction). `issue_category` (lost_parcel: 23.8%) dominates feature importance.
- **Classification:** Random Forest F1 0.557 / PR-AUC 0.573 (escalated class) vs. DummyClassifier F1 0.0. `previous_ticket_count` (33.0%) is the dominant feature — notably different from the regression model's top feature, indicating the two tasks have substantially different drivers.
- Controlled `max_depth` experiment (6/12/20/None) shows MAE and RMSE disagree on the optimum; `max_depth=12` was retained, explicitly justified by the business use case favoring MAE.

### 5. Model evaluation and tuning
Same files as above, plus the cross-validation/subgroup section at the bottom of `02_classification_notes.md`.

- 5-fold stratified CV: mean macro F1 0.6601, std 0.0018 — stable across folds.
- Full probability-threshold table (0.3–0.7) enabling the business to pick an operating point by senior-agent review capacity.
- Held-out test set confirms validation-set metrics to within 0.001 (F1 0.5570 → 0.5577), demonstrating no overfitting to the validation set despite extensive analysis against it.
- Demographic subgroup fairness across 10 UK regions + "Unknown": F1 spread of just 0.014 (0.5505–0.5644) — no region systematically underserved.

### 6. Unsupervised learning and clustering
`starter_code/02_ml/03_clustering_notes.md`

KMeans (k=2–6) selected via silhouette score (sampled at 20k customers); k=4 chosen as the practical balance point given uniformly low silhouette scores (0.11–0.16, indicating a continuum rather than sharp clusters, a property of the data, not a modeling failure). Four segments described cautiously, without causal claims. Cluster 3 (250 customers, extreme resolution times) is independently validated against the same `open_after_7d` anomaly identified in the Phase 2 data audit, the clustering algorithm rediscovered a known structural pattern from aggregated numeric features alone.

### 7. Deep learning
`starter_code/03_deep_learning/01_deep_learning_notes.md`

A small feedforward network (61→64→32→1) trained on the same feature set/split as the Phase 4 regression task, for direct comparability. Learning-rate comparison (0.01/0.001/0.0001) shows all three converge to ~3.50 MAE, indicating a genuine architecture/feature-set ceiling rather than a tuning artifact. Marginally beats the Random Forest (3.50 vs. 3.76) but plateaus almost immediately, Random Forest is judged the more practical choice given comparable accuracy at lower complexity. Checkpoint saved (`outputs/03_deep_learning/tabular_nn_checkpoint.pt`); `torch.manual_seed(42)` for reproducibility.

### 8. Computer vision
`starter_code/04_computer_vision/01_vision_notes.md`

Basic 3-layer CNN trained from scratch reaches 100% test accuracy on 480 synthetic return images (4 balanced classes). ResNet18 transfer learning (frozen backbone) scores 96%, counter-intuitively lower, explained by a mismatch between ImageNet's natural-photo pretrained features and this task's synthetic visual cues, combined with the task's evident simplicity leaving little room for transfer learning's usual advantage. Full per-class precision/recall/F1 and confusion-matrix analysis included. Both results are explicitly scrutinized rather than taken at face value, and framed as a property of this synthetic dataset rather than a claim of real-world generalization.

### 9. NLP, attention and transformers
`starter_code/05_nlp/01_classical_nlp_notes.md`, `02_semantic_search_notes.md`, `starter_code/06_transformers/01_attention_notes.md`, `02_pretrained_pipelines_notes.md`, `03_finetuning_notes.md`

- BoW vs. TF-IDF: identical macro F1 (0.9925), explained by the dataset's templated phrase structure. All 16 validation errors traced to blank-description tickets, a documented data quality issue, not a model limitation.
- Semantic search vs. keyword matching: a paraphrased query with no shared vocabulary is correctly retrieved by embeddings (damaged_item, similarity 0.57–0.60) while TF-IDF fails outright (wrong category, similarity 0.33).
- **Attention worked example (mandatory):** from-scratch Q/K/V self-attention over a 9-token support message, with sinusoidal positional encoding and a full attention heatmap (`outputs/06_transformers/attention_heatmap.png`), explicitly contrasted against TF-IDF's order-blind, uniform-weight representation of the same message.
- Pretrained pipelines (sentiment + zero-shot classification) applied without fine-tuning, both performing strongly (>99% sentiment confidence, 88–93% zero-shot confidence) on damage-related tickets.
- **Fine-tuning (mandatory, Session 22):** DistilBERT fine-tuned on a 1,500-example sample (CPU-only, compute trade-off explicitly justified), achieving macro F1 1.0000 vs. TF-IDF's 0.9925. The marginal gain is explicitly judged not to justify the added training cost for this templated dataset, with the real business case for fine-tuning framed around messier, real-world text where TF-IDF would degrade faster.

### 10. Prompt engineering and LLM application performance
`starter_code/07_llm/01_llm_notes.md`

Pydantic-validated structured ticket extraction; 5/5 hand-written regression tests passing on the paid model. **20-request benchmark (mandatory, Session 25):** mean latency 541ms (range 328–890ms), 1,568 total tokens, cost $0.000594 — extrapolated to ~$0.30/day at 10,000 tickets/day. Explicit trade-off discussion: free-tier `google/gemma-4-26b-a4b-it:free` hit persistent 429 rate limits under sustained load even with exponential backoff, motivating a switch to a low-cost paid tier; latency variance (2.7× range) is flagged as a reliability consideration for production timeout/fallback design.

### 11. RAG
`starter_code/08_rag/01_rag_notes.md`, `starter_code/10_evaluation/01_rag_evaluation_notes.md`

40 policy documents chunked (300-word/50-word overlap; effectively one chunk per document given document length), embedded with `all-MiniLM-L6-v2`, retrieved via cosine similarity, answered by `google/gemini-2.5-flash` with mandatory source citation and explicit no-answer instructions. Chunking comparison (300-word vs. 40-word) shows finer chunking improves retrieval precision at a 3× storage cost, judged a close-to-a-wash trade-off for this short-document knowledge base. **Evaluation on 15 independent questions:** 13/15 (87%) confidently correct and cited; 2/15 correctly refused rather than fabricated; average top retrieval score 0.547; no hallucinations observed. A clear retrieval-score gap between answerable (0.50–0.70) and out-of-scope (0.17–0.18) questions is used to justify a proposed 0.5 human-review threshold, further corroborated by a live post-deployment test (see Section 16).

### 12. Agent and tool calling
`starter_code/09_agents/01_agent_notes.md`

Five local tools (`lookup_order`, `lookup_customer`, `check_return_eligibility`, `calculate_refund`, `search_policy`) against the supplied 15,000-order operational store, with in-memory caching per the starter code's guidance. Design principle: the LLM only performs intent extraction; all approval-threshold logic is enforced in code ("LLM proposes, code disposes"). The £100 frontline approval threshold triggers on whichever is higher, the tool's calculated amount or the customer's requested amount, closing a gap that would otherwise let an inflated demand through unchecked (test case A03). All 8 supplied test cases pass, covering missing-information handling (A04), refused account changes (A06), avoided false action confirmation (A07), and safety escalation (A08). Documented limitation: intent routing depends on LLM extraction quality, and a larger adversarial test set is recommended for production.

### 13. Evaluation, guardrails and testing
`starter_code/10_evaluation/01_rag_evaluation_notes.md`, `02_prompt_evaluation_notes.md`, `tests/starter_tests/`, `tests/extended_tests/test_escalation_endpoint.py`

15-question RAG evaluation (above) plus a 10-case prompt evaluation: 10/10 schema-complete, 3/3 safety-critical cases correctly handled (refused unsafe secret collection; correctly distinguished a proposed refund from a completed action; avoided an unsupported compatibility claim). A methodology note documents a genuine evaluation-tooling bug caught and fixed: an early automated refusal-detection pass over-counted refusals via a naive keyword check, corrected to check only answer-initial phrasing.

**Test suite integrity fix.** During final review, 4 of the 10 starter test files (`test_api_health.py`, `test_api_schema_validation.py`, `test_rag_contract.py`, `test_vision_contract.py`) were found to import the original stub API (`app/api/main.py`, where every endpoint raises `HTTPException(501)`) rather than the real implementation at `starter_code/10_api/main.py`. This meant these tests were passing or skipping against dead code, not validating actual system behavior. Three were corrected to load the real app via `importlib.util` (matching the pattern already used by the agent/structured-output tests) and target the real endpoint paths and status codes (e.g. `/ask/policy`'s manual empty-question check returns 400, not the stub's 422); `test_vision_contract.py` was deliberately left as-is, since the real API does not expose a vision endpoint (CNN evaluation is validated via direct module testing in Phase 8 instead). A second issue surfaced once the tests targeted the real app: `TestClient` was being used without triggering FastAPI's `startup` event, so the classifier and RAG index never loaded, causing spurious 503s — fixed by explicitly entering the `TestClient` context (`client.__enter__()`). A new file, `tests/extended_tests/test_escalation_endpoint.py`, adds genuinely new coverage for `/predict/escalation` (valid-request probability bounds, missing-field rejection, and a directional sanity check that a high-priority/high-ticket-history profile scores at least as high as a low-risk one), an endpoint with no prior test coverage at all. All 16 active tests pass in CI after also adding the previously-missing `pyarrow` dependency to `requirements-ci.txt` (needed once these tests began actually loading the parquet dataset via the real startup path).

### 14. Responsible AI and governance
`starter_code/10_evaluation/03_responsible_ai_risk_register.md`

A seven-row risk register (privacy, subgroup fairness, hallucination, unsafe actions, content safety, auditability, human oversight) built directly from project evidence rather than generic claims, each row cites a specific test result (e.g. the Section 5 demographic fairness numbers, the Section 11 RAG refusal behavior, the Section 12 agent approval routing). Residual risks are stated honestly (e.g. no persistent audit log currently implemented; fixed £100 threshold is a business policy choice, not a technical one) rather than presented as fully resolved.

### 15. API/UI integration
`starter_code/10_api/01_api_notes.md`, `starter_code/11_ui/01_ui_notes.md`

FastAPI service with three endpoints (`/health`, `/predict/escalation`, `/ask/policy`), Pydantic request/response validation, and controlled error responses (400 for empty questions, 503 for a not-yet-loaded model) rather than unhandled crashes. Two-tab Streamlit UI calls the API over HTTP (not direct model import), mirroring a real frontend/backend split and exercising the API contract itself. Both interfaces independently reproduce the same RAG answer quality and escalation-scoring behavior validated in earlier phases, confirming consistency across every exposed surface.

### 16. Deployment, CI, reproducibility and monitoring
`deployment/Dockerfile`, `.github/workflows/tests.yml`, `starter_code/12_monitoring/01_monitoring_notes.md`, `monitoring/monitoring_summary_template.csv`

- **Docker:** `Dockerfile` builds and runs the FastAPI service; `CMD` uses `--app-dir` rather than a dotted import path, since `10_api` is not a valid Python module name.
- **CI:** `.github/workflows/tests.yml` passes, with `LLM_API_KEY`/`MODEL_NAME` injected as repository secrets (required because `rag_pipeline.py` instantiates its LLM client at import time).
- **Bug found and fixed during container verification:** a live end-to-end `/ask/policy` test initially failed with an OpenRouter `401 Missing Authentication header` error. Root cause: a leading space in `.env` (`LLM_API_KEY= sk-...`) was preserved literally by `python-dotenv`, producing a malformed `Authorization` header. Diagnosed via a length/preview check inside the running container (74 vs. expected 73 characters) rather than assuming a wrong key or provider misconfiguration. Fixed with no rebuild required. Full write-up in `01_monitoring_notes.md`.
- **Monitoring:** `monitoring.py`'s `record_request`/`record_rag_outcome`/`record_agent_tool_failure` functions were implemented (starter code shipped as `NotImplementedError` stubs) and used to backfill a summary from Phase 9/10/12 evaluation results plus one live container test, producing the required fields (request count, average latency, failure count, RAG no-answer rate, agent tool-failure count, model version, prompt version). This reflects a documented limitation: no persistent live request logging was implemented for this project's scope (see the Auditability row in the risk register), so the summary is a validated snapshot rather than a live production feed. Output: `monitoring/monitoring_summary_template.csv`, matching both the supplied starter filename and the path asserted by `tests/starter_tests/test_data_structure.py`.

### 17. Code quality, documentation and engineering reasoning
This README, plus per-phase notes files throughout `starter_code/*/`.

Each phase's notes file follows a consistent structure (setup → results → interpretation → limitations), documents controlled single-variable experiments where tuning is discussed (Phase 4's `max_depth` sweep, Phase 6's learning-rate sweep), and states assumptions and trade-offs explicitly rather than only reporting favorable outcomes (e.g. the deep-learning-vs-Random-Forest and transfer-learning-vs-CNN sections both conclude that the simpler approach was the better practical choice, despite being the "less advanced" method).

## Known Limitations (Project-Wide)

- **Synthetic data separability.** Several components (image classification, TF-IDF text classification, DistilBERT fine-tuning) reach near-perfect or perfect scores. This is explicitly attributed throughout to the synthetic dataset's templated/highly-separable structure, not claimed as evidence of real-world generalization.
- **No persistent request logging.** The monitoring summary is backfilled from evaluation-phase results and one live container test, not a running production log (see Section 16 and the Responsible AI risk register's Auditability row).
- **Fixed business thresholds.** The £100 agent-approval limit and the proposed 0.5 RAG human-review threshold are both currently hard-coded design choices, not values that adapt per customer/order risk profile, flagged as an intentional scope simplification in both the agent notes and the risk register.
- **Subgroup fairness scope.** Demographic fairness analysis (Section 5) covers `customer_region` only; other attributes (age band, language, accessibility flag) are not yet analyzed, and the risk register recommends extending this before production use.
- **LLM-dependent intent routing.** The agent's tool selection depends on LLM extraction quality from free text; all 8 supplied test cases pass, but a larger adversarial test set is recommended before production deployment.

## Repository Structure

```
starter_code/       # All phase implementations and notes, numbered by phase
knowledge_base/      # 40 policy documents used by the RAG pipeline
evaluation/          # Supplied test cases (agent, prompt, RAG questions)
data/                # Primary dataset, agent store, images, subsets
deployment/          # Dockerfile, docker-compose.yml
monitoring/          # Monitoring summary output
tests/               # Automated contract tests (starter + extensions)
.github/workflows/   # CI pipeline (tests.yml)
```

## Demo

A short screen-recorded walkthrough of the working system is available here (unlisted YouTube video): **https://youtu.be/XpW0GlPVRz4**

The recording covers: the Streamlit Escalation Risk Predictor tab returning a live probability score for a sample ticket, the Policy Assistant tab returning a cited RAG answer to a policy question, and a walkthrough of the Docker container running the same system end-to-end via the FastAPI `/docs` interface.