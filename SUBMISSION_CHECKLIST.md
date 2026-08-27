# Submission Checklist

Before submission, confirm that your repository contains evidence for every item below.

- [ ] Problem statement, target(s), business decision(s), prediction timing and success metrics.
- [ ] Reproducible data audit/cleaning across the 1,000,000-row primary Parquet dataset.
- [ ] Missingness, duplicates, inconsistent categories, outliers, imbalance and leakage analysis.
- [ ] EDA with useful visualisations and defensible engineered features.
- [ ] Regression and classification baselines, model comparisons, suitable metrics and business interpretation.
- [ ] Cross-validation, tuning and at least one decision-threshold analysis where appropriate.
- [ ] Clustering with validation and cautious personas/interpretation.
- [ ] Tabular neural network with train/validation evidence and checkpointing.
- [ ] Basic CNN and transfer-learning comparison with vision error analysis.
- [ ] BoW/TF-IDF comparison, NLP classifier and semantic search.
- [ ] **Session 20:** worked/visual attention explanation covering query, key, value, self-attention and positional information, contrasted with TF-IDF.
- [ ] Pretrained transformer pipeline evidence.
- [ ] **Session 22:** completed fine-tuning/adaptation of a small Hugging Face text classifier using the supplied 10,000-row subset (or a justified smaller training sample), evaluated and compared with TF-IDF.
- [ ] Prompt library and validated structured outputs.
- [ ] **Session 25:** benchmark of at least 20 LLM requests with average latency and estimated token/cost plus a documented quality/latency/cost/context/streaming/UX trade-off.
- [ ] RAG with chunking comparison, source display, evaluation and no-answer handling.
- [ ] Agent using local tools, missing-information handling, tool-failure handling and human approval gates.
- [ ] Evaluation/guardrails plus responsible-AI risk register and subgroup analysis.
- [ ] Working API/UI integration and controlled validation errors.
- [ ] Docker build/run instructions.
- [ ] `.github/workflows/tests.yml` retained and CI evidence included.
- [ ] Automated tests extended beyond starter contracts for your implementation.
- [ ] Observable monitoring output with request count, average latency, failures, RAG no-answer rate, agent tool failures, model version and prompt version.
- [ ] README/architecture/deployment/monitoring documentation and demo evidence.
- [ ] No instructor-only files, hidden labels, hard-coded expected answers or irreversible external actions.
