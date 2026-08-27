# Submission Guide

Submit a single GitHub-style project repository or ZIP containing your implementation, documentation and evidence. Do not submit the supplied raw image files or million-row dataset back to the marker unless your instructor specifically asks; your code should reference the supplied data layout.

## Minimum evidence
- README with setup, architecture, run instructions and system limitations.
- Reproducible notebooks/scripts for data, models and evaluation.
- Saved evaluation tables/figures sufficient to verify claims.
- Attention worked example/visualisation and transformer fine-tuning comparison.
- LLM benchmark table covering at least 20 requests.
- RAG and agent evaluation results.
- API/UI run instructions.
- Docker instructions and CI run evidence.
- Monitoring summary/log evidence.
- Responsible-AI risk register.
- Short demo or presentation evidence as requested by the instructor.

## Reproducibility
Use the pinned `requirements.txt` baseline unless your instructor authorises changes. If you change a version, document why. Keep API keys out of source control and use `.env`/environment variables.

## Assessment integrity
You may use documentation and normal development tools, but you must understand and be able to explain all submitted code. The viva may ask you to justify model choices, metrics, attention behaviour, fine-tuning decisions, RAG retrieval, agent tool selection, CI failures, and monitoring signals.
