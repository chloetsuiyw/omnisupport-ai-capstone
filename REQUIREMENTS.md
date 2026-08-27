# Technical Requirements

## Allowed course technologies
Use only technologies and depth taught in the 30-session course: Python, NumPy, Pandas, Matplotlib/Seaborn, Scikit-learn, PyTorch or TensorFlow, Hugging Face Transformers, a course-approved LLM API, FAISS or Chroma, LangChain or LlamaIndex, Streamlit or Gradio, FastAPI, Docker, GitHub, and basic CI/CD/logging/monitoring concepts.

The supplied baseline is pinned in `requirements.txt` for reproducibility. Python 3.12 is the reference runtime used by the Docker and CI starter files.

## Mandatory engineering evidence
- Reproducible loading of all ten primary Parquet shards and a documented data-cleaning pipeline.
- Explicit prediction-time leakage audit.
- Classical regression and classification with baselines and comparison.
- Cross-validation and tuning for at least one classical pipeline.
- At least one clustering analysis with cautious interpretation.
- A small tabular neural-network experiment with a recorded training/validation loop.
- Basic CNN plus transfer-learning comparison on the supplied synthetic image set.
- Classical NLP text classification plus embedding-based semantic search.
- **Session 20 attention evidence:** explain query, key, value, self-attention, and positional information using one customer-support message. Include either a worked numerical example or a visualisation, then contrast what attention represents with TF-IDF.
- Pretrained transformer pipeline use.
- **Session 22 mandatory fine-tuning:** fine-tune/adapt a small Hugging Face text classifier using the supplied `data/subsets/transformer_finetune_10000.parquet` dataset (you may use a smaller training sample only when you justify the hardware limitation). Compare the fine-tuned model with your TF-IDF baseline on an appropriate held-out set. This task is compulsory.
- Prompt library and validated structured output.
- **Session 25 LLM application benchmark:** run at least 20 representative LLM requests and report average latency plus estimated token usage/cost. Discuss at least one trade-off involving response quality, latency, cost, context-window use, streaming, API-key handling, or UX.
- RAG assistant over the supplied policy corpus with visible sources.
- Controlled AI agent using the supplied local `data/agent_store/` data and human-approval gates.
- Evaluation, guardrails, privacy/fairness analysis, and a risk register.
- FastAPI and/or UI integration plus Docker packaging.
- **CI/CD evidence:** keep `.github/workflows/tests.yml` working and show that syntax/tests/basic validation run in CI. Extend tests for the components you implement.
- **Observable monitoring evidence:** produce a monitoring output such as `monitoring/monitoring_summary.csv` or an equivalent small dashboard/log summary containing, at minimum: request count, average latency, failure count, RAG no-answer rate, agent tool-failure count, model version, and prompt version.

## Data boundaries
The ten files matching `data/raw/support_records_part_*.parquet` are the **primary dataset** and must total exactly 1,000,000 records. `dataset_preview.csv`, the transformer subset, images, agent-store CSV files, and evaluation data are supporting/derived assets and must not be added to the primary row count.

## Prohibited shortcuts
Do not submit tutorial clones, generated notebooks you cannot explain, hidden hard-coded answers to evaluation cases, instructor-only files, or code that performs irreversible external actions. Do not use post-outcome leakage fields for prediction unless the task explicitly studies leakage.
