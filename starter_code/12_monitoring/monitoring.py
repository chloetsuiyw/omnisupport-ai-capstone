"""Minimal monitoring starter for Session 30.

Keep this course-sized: collect observable counters/latency/version fields and
export a compact CSV/log summary. Advanced observability platforms are not
required.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import csv


@dataclass
class MonitoringSummary:
    request_count: int = 0
    average_latency_ms: float = 0.0
    failure_count: int = 0
    rag_no_answer_rate: float = 0.0
    agent_tool_failure_count: int = 0
    model_version: str = "TODO"
    prompt_version: str = "TODO"


def record_request(summary: MonitoringSummary, latency_ms: float, success: bool = True) -> None:
    """Update summary with one observed request.

    Recomputes the running mean latency incrementally (rather than storing
    every individual latency value), and increments failure_count on
    unsuccessful requests. RAG-specific and agent-specific counters
    (rag_no_answer_rate, agent_tool_failure_count) are updated separately
    via record_rag_outcome / record_agent_tool_failure, since not every
    request touches the RAG or agent components.
    """
    total_latency = summary.average_latency_ms * summary.request_count
    summary.request_count += 1
    total_latency += latency_ms
    summary.average_latency_ms = total_latency / summary.request_count
    if not success:
        summary.failure_count += 1


def record_rag_outcome(summary: MonitoringSummary, answered: bool, total_rag_requests: int) -> None:
    """Update the running rag_no_answer_rate given one more RAG request outcome.

    total_rag_requests is the running count of RAG-specific requests seen so
    far (may differ from summary.request_count, since not all requests are
    RAG questions).
    """
    prior_no_answer_count = summary.rag_no_answer_rate * (total_rag_requests - 1)
    if not answered:
        prior_no_answer_count += 1
    summary.rag_no_answer_rate = prior_no_answer_count / total_rag_requests


def record_agent_tool_failure(summary: MonitoringSummary) -> None:
    """Increment the count of agent tool-call failures."""
    summary.agent_tool_failure_count += 1


def export_summary(summary: MonitoringSummary, output_path: str | Path):
    """Small supplied utility: export a completed summary object to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = asdict(summary)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    # Populate a summary from real evaluation results gathered across
    # Phases 9, 10, 12a, 12b, and this phase's live container test,
    # rather than from a live request log (no persistent logging was
    # implemented for this project's scope — see Phase 14 notes,
    # Auditability risk row in 03_responsible_ai_risk_register.md).
    summary = MonitoringSummary(
        model_version="google/gemini-2.5-flash",
        prompt_version="rag_policy_prompt_v1",  # update to your actual prompt version tag if you track one
    )

    # Phase 9: 20-request LLM benchmark (individual latencies not retained,
    # only mean/range — replay using the reported mean as an approximation)
    for _ in range(20):
        record_request(summary, latency_ms=541)

    # Live container test (Phase 14): 1 failed request (pre-fix, .env
    # whitespace bug) + 1 successful request (post-fix)
    record_request(summary, latency_ms=0, success=False)  # pre-fix 500 error, latency not meaningful
    record_request(summary, latency_ms=1200, success=True)  # post-fix successful /ask/policy call (approx, not precisely timed)

    # Phase 12a: RAG evaluation, 15 questions, 2 refusals ("no answer")
    for i in range(15):
        answered = i not in (3, 11)  # RAG04 and RAG12 were the 2 appropriate refusals
        record_rag_outcome(summary, answered=answered, total_rag_requests=i + 1)

    # No agent tool-call failures were observed in Phase 11/12 testing
    # (agent tests A03/A06/A07/A08 correctly routed to human approval,
    # which is expected behavior, not a tool failure)

    export_summary(summary, "monitoring/monitoring_summary_template.csv")
    print("Monitoring summary exported.")