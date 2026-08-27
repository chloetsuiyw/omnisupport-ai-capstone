# Project Tasks

## Phase 1 - Business Understanding and Problem Framing
Define users, business decisions, measurable success metrics, targets, prediction timing, risks, and system boundaries.

## Phase 2 - Data Audit, Cleaning and EDA
Profile the million-row Parquet extract, quantify missing values/blanks, duplicates, inconsistent categories, outliers and imbalance, clean reproducibly, visualise important relationships, and document assumptions.

## Phase 3 - Feature Engineering
Create defensible numerical, categorical, date/time, interaction, and text-derived features while preventing prediction-time leakage.

## Phase 4 - Classical ML
Build at least one regression task and one classification task. Compare baselines and multiple suitable models, tune one pipeline, examine relevant probability thresholds, and interpret results in business terms.

## Phase 5 - Unsupervised Learning
Create a meaningful customer or ticket segmentation, validate cluster quality, and write cautious personas without claiming causality.

## Phase 6 - Deep Learning
Implement a small neural network for a suitable tabular task; track train/validation loss, compare optimisation settings, discuss overfitting, and save a checkpoint.

## Phase 7 - Computer Vision
Train a basic CNN on the supplied return images, then compare it with transfer learning. Include augmentation, a confusion matrix, class-aware metrics, and error examples.

## Phase 8 - NLP, Attention and Transformers
1. Compare bag-of-words and TF-IDF and build a classical ticket text classifier.
2. Create embeddings for semantic search and analyse retrieval examples.
3. **Attention task (mandatory):** for one support message, explain query, key, value, self-attention and positional information. Include a worked example or visualisation and contrast the representation with TF-IDF.
4. Use pretrained Hugging Face transformer pipelines for at least two suitable NLP tasks.
5. **Fine-tuning task (mandatory):** use `data/subsets/transformer_finetune_10000.parquet` to fine-tune/adapt a small text-classification transformer. Evaluate it on held-out examples and compare it with the TF-IDF baseline. You may reduce the training sample only if you document a genuine compute limitation and still complete a real fine-tuning run.

## Phase 9 - Prompt Engineering, Structured Outputs and LLM App Performance
Design reusable prompts for extraction/classification/summarisation, validate structured outputs with Pydantic, and run prompt regression tests. Then execute at least 20 representative LLM requests and record latency plus estimated tokens/cost. Explain at least one engineering trade-off involving quality, latency, cost, context-window use, streaming, API-key handling, or UX.

## Phase 10 - Semantic Search and RAG
Ingest the policy corpus, test at least two chunking settings, create embeddings/vector storage, retrieve evidence, generate grounded answers with source display, and evaluate the supplied questions. Include unsupported/conflicting-policy cases and an abstention/no-answer strategy.

## Phase 11 - AI Agent and Tool Calling
Implement safe local tools against `data/agent_store/orders.csv`, `customers.csv`, and `returns.csv`; define tool schemas; build a controlled agent loop; handle missing inputs/tool failures; and require human approval for sensitive actions. Follow the supplied agent test cases, including cases where the correct behaviour is to request missing information rather than call a tool.

## Phase 12 - Evaluation, Guardrails and Responsible AI
Create an evaluation rubric, hallucination/failure checks, subgroup analysis, privacy/safety guardrails, regression tests, and a responsible-AI risk register.

## Phase 13 - API/UI Integration
Expose selected functionality through FastAPI and/or a small Streamlit/Gradio stakeholder demo. Validate request schemas and return controlled errors for unsupported or unsafe requests.

## Phase 14 - Docker, CI/CD, Logging and Monitoring
Containerise the prototype, add useful structured logs, keep `.github/workflows/tests.yml` operational, and extend the starter test contracts. Produce an observable monitoring artifact containing request count, average latency, failures, RAG no-answer rate, agent tool failures, model version and prompt version. Document versioning, feedback loops and incident response.

## Phase 15 - Final Engineering Report and Demonstration
Submit a coherent repository, architecture explanation, evaluation evidence, deployment/CI instructions, monitoring evidence, risks, and a short demo that explains engineering trade-offs.
