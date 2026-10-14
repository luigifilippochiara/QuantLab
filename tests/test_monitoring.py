"""Monitoring: the JSONL event log and the dashboard aggregations."""

from pathlib import Path

import pytest

from quantlab.monitoring import (
    capability_counts,
    guardrail_counts,
    kpis,
    log_event,
    read_events,
    recent_errors,
    run_outcomes,
    stage_latency,
)


@pytest.fixture
def events(tmp_path: Path) -> list[dict]:
    path = tmp_path / "events.jsonl"
    # Run 1: blocked at validation, then edited and completed.
    log_event(path, "r1", "request_received")
    log_event(
        path,
        "r1",
        "plan_generated",
        stage="planning",
        latency_ms=100.0,
        details={"mode": "cached", "from_cache": True},
    )
    log_event(path, "r1", "plan_validated", stage="validation", latency_ms=2.0)
    log_event(
        path,
        "r1",
        "validation_issue",
        details={"code": "G03", "name": "INVALID_PARAM", "severity": "BLOCK"},
    )
    log_event(path, "r1", "run_failed", details={"status": "blocked", "error": "G03"})
    log_event(path, "r1", "step_completed", details={"capability": "load_series"})
    log_event(path, "r1", "step_completed", details={"capability": "resample"})
    log_event(path, "r1", "execution_finished", stage="execution", latency_ms=50.0)
    log_event(path, "r1", "run_completed")
    # Run 2: a step fails during execution.
    log_event(path, "r2", "request_received")
    log_event(
        path,
        "r2",
        "plan_generated",
        stage="planning",
        latency_ms=300.0,
        details={"mode": "live", "from_cache": False},
    )
    log_event(path, "r2", "step_failed", details={"capability": "resample", "error": "boom"})
    log_event(
        path,
        "r2",
        "guardrail_triggered",
        details={"code": "R02", "name": "INSUFFICIENT_DATA", "severity": "BLOCK"},
    )
    log_event(path, "r2", "run_failed", details={"status": "failed", "error": "boom"})
    # Run 3: still waiting for approval.
    log_event(path, "r3", "request_received")
    log_event(path, "r3", "plan_generated", details={"mode": "mock"})
    return read_events(path)


def test_log_event_rejects_unknown_types(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        log_event(tmp_path / "e.jsonl", "r", "made_up_event")


def test_read_events_skips_broken_lines(tmp_path: Path) -> None:
    path = tmp_path / "e.jsonl"
    log_event(path, "r", "request_received")
    with path.open("a") as handle:
        handle.write('{"truncated": \n')
    log_event(path, "r", "run_completed")
    assert [e["event_type"] for e in read_events(path)] == ["request_received", "run_completed"]
    assert read_events(tmp_path / "missing.jsonl") == []


def test_last_terminal_event_decides_the_outcome(events: list[dict]) -> None:
    assert run_outcomes(events) == {"r1": "completed", "r2": "failed", "r3": "in_progress"}


def test_kpis(events: list[dict]) -> None:
    k = kpis(events)
    assert (k["n_runs"], k["n_completed"], k["n_failed"], k["n_in_progress"]) == (3, 1, 1, 1)
    assert k["success_rate"] == 0.5
    assert k["cache_hit_rate"] == 0.5  # one cached hit out of two non-mock plans


def test_stage_latency(events: list[dict]) -> None:
    rows = {row["stage"]: row for row in stage_latency(events)}
    assert rows["planning"]["count"] == 2
    assert rows["planning"]["mean_ms"] == 200.0
    assert rows["planning"]["p95_ms"] == 290.0  # linear interpolation between 100 and 300
    assert rows["narration"]["count"] == 0 and rows["narration"]["mean_ms"] is None


def test_capability_guardrail_and_error_tables(events: list[dict]) -> None:
    assert capability_counts(events) == {"resample": 2, "load_series": 1}
    codes = {(row["code"], row["severity"]): row["count"] for row in guardrail_counts(events)}
    assert codes == {("G03", "BLOCK"): 1, ("R02", "BLOCK"): 1}
    errors = recent_errors(events)
    assert errors[0]["run_id"] == "r2" and errors[0]["event_type"] == "run_failed"
    assert len(errors) == 3
