# OmniSupport AI — Final AI Engineering Capstone

You are the AI Engineer on a retail support modernisation project. Build one connected system that helps an online retailer analyse support operations, predict outcomes, classify and search ticket text, analyse return images, answer policy questions with RAG, execute controlled support tools through an agent, and expose selected capabilities through an API/UI.

This is an assessment repository, not a solved tutorial. Core model, LLM, RAG, agent, evaluation and integration functions are intentionally incomplete.

## Supplied assets
- **Primary data:** 10 Parquet shards × 100,000 rows = exactly **1,000,000 records** in `data/raw/`.
- CSV preview for quick inspection.
- 480 synthetic return/damage images with labels.
- 10,000-row text subset for the **mandatory Session 22 Hugging Face fine-tuning task**.
- Local agent operational store with 15,000 orders plus customer/return data.
- 40 policy/SOP knowledge-base documents.
- Prompt, RAG and agent evaluation cases.
- Starter code, API/UI scaffolding, Docker files, CI workflow, monitoring template and test contracts.

## Required additions in this revised pack
Four areas are explicitly compulsory because they correspond to taught sessions that can otherwise be under-assessed:
1. **Attention (Session 20):** Q/K/V, self-attention and positional information with a worked example/visualisation and a TF-IDF contrast.
2. **Transformer fine-tuning (Session 22):** a real small-model fine-tuning run and comparison with TF-IDF.
3. **LLM application engineering (Session 25):** benchmark at least 20 requests for latency and estimated token/cost, then discuss engineering trade-offs.
4. **Production evidence (Session 30):** CI plus observable monitoring outputs, not only a written monitoring plan.

Read `PROJECT_BRIEF.pdf`, `PROJECT_TASKS.md`, `REQUIREMENTS.md`, and `SUBMISSION_CHECKLIST.md` before starting.
