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


def record_request(*args, **kwargs):
    """TODO: record request latency/success and component-specific events."""
    raise NotImplementedError("Student task")


def export_summary(summary: MonitoringSummary, output_path: str | Path):
    """Small supplied utility: export a completed summary object to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = asdict(summary)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
