# Monitoring Evidence

Session 30 requires an **observable artifact**, not only a written monitoring plan. Create `monitoring_summary.csv` (or a small equivalent dashboard/log-derived table) from representative local requests.

Minimum fields:
- request_count
- average_latency_ms
- failure_count
- rag_no_answer_rate
- agent_tool_failure_count
- model_version
- prompt_version

Explain how each signal would help detect quality, reliability or version regressions. Keep the implementation within the basic logging/monitoring level taught in the course.
