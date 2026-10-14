"""Monitoring: an append-only JSONL event log and the aggregations behind the dashboard.

Role in the architecture: every stage of every run (and every replay) appends
one JSON object per line to runs/events.jsonl. Nothing is ever rewritten, so
the log is a simple audit trail and is trivial to ship to a real log platform
later. The functions at the bottom turn the events into the numbers shown on
the Monitoring page: run outcomes, latency per stage, capability usage,
guardrail activity, recent errors and the plan-cache hit rate.

The event types are listed in EVENT_TYPES below: the spec's list plus
"plan_validated" and "execution_finished", which carry the validation and
execution stage latencies.
"""

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

EVENT_TYPES = (
    "request_received",
    "plan_generated",
    "plan_failed",
    "plan_validated",
    "validation_issue",
    "approval_granted",
    "execution_started",
    "step_completed",
    "step_failed",
    "guardrail_triggered",
    "execution_finished",
    "narrative_checked",
    "run_completed",
    "run_failed",
    "replay_completed",
)

# Which event carries the latency of which pipeline stage.
STAGE_EVENTS = {
    "planning": ("plan_generated", "plan_failed"),
    "validation": ("plan_validated",),
    "execution": ("execution_finished",),
    "narration": ("narrative_checked",),
}


def utc_now() -> str:
    """Current UTC time as an ISO-8601 string with milliseconds, e.g. 2026-10-14T19:24:42.123Z."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def log_event(
    path: Path,
    run_id: str,
    event_type: str,
    *,
    stage: str | None = None,
    latency_ms: float | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one event to the JSONL log.

    Args:
        path: the events file (created with its folder if missing).
        run_id: the run the event belongs to.
        event_type: one of EVENT_TYPES.
        stage: pipeline stage ("planning", "validation", "execution", "narration").
        latency_ms: duration of that stage, if the event closes one.
        details: extra fields (capability, issue code, severity, error message...).

    Returns:
        The event dict that was written.

    Raises:
        ValueError: for an unknown event type (catches typos at the call site).
    """
    if event_type not in EVENT_TYPES:
        raise ValueError(f"unknown event type {event_type!r}")
    event = {
        "timestamp": utc_now(),
        "run_id": run_id,
        "event_type": event_type,
        "stage": stage,
        "latency_ms": latency_ms,
        "details": details or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, default=str) + "\n")
    return event


def read_events(path: Path) -> list[dict[str, Any]]:
    """Read all events, oldest first. Skips lines that are not valid JSON.

    Args:
        path: the events file.

    Returns:
        List of event dicts (empty if the file does not exist yet).
    """
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a half-written line (e.g. after a crash) must not break the dashboard
    return events


# ---------------------------------------------------------------------------
# Aggregations for the dashboard
# ---------------------------------------------------------------------------


def run_outcomes(events: list[dict[str, Any]]) -> dict[str, str]:
    """Final status of each run: "completed", "blocked", "failed" or "in_progress".

    A run's status is taken from its LAST run_completed / run_failed event, so a
    plan that was blocked, then edited and executed successfully counts as completed.

    Args:
        events: all events.

    Returns:
        run_id -> status, for every run that has a request_received event.
    """
    outcomes: dict[str, str] = {}
    for event in events:
        run_id, kind = event["run_id"], event["event_type"]
        if kind == "request_received":
            outcomes.setdefault(run_id, "in_progress")
        elif kind == "run_completed":
            outcomes[run_id] = "completed"
        elif kind == "run_failed":
            outcomes[run_id] = event["details"].get("status", "failed")
    return outcomes


def kpis(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Headline numbers: run counts and rates, replay results, plan-cache hit rate.

    Args:
        events: all events.

    Returns:
        Dict with n_runs, n_completed, n_blocked, n_failed, n_in_progress,
        success_rate, blocked_rate, failed_rate (shares of finished runs, None
        if no run finished), n_replays, replay_match_rate and cache_hit_rate
        (share of non-mock plans served from the cache, None if there were none).
    """
    statuses = Counter(run_outcomes(events).values())
    finished = statuses["completed"] + statuses["blocked"] + statuses["failed"]

    def share(count: int, total: int) -> float | None:
        return round(count / total, 4) if total else None

    replays = [e for e in events if e["event_type"] == "replay_completed"]
    llm_plans = [
        e
        for e in events
        if e["event_type"] == "plan_generated" and e["details"].get("mode") != "mock"
    ]
    cache_hits = sum(1 for e in llm_plans if e["details"].get("from_cache"))
    return {
        "n_runs": sum(statuses.values()),
        "n_completed": statuses["completed"],
        "n_blocked": statuses["blocked"],
        "n_failed": statuses["failed"],
        "n_in_progress": statuses["in_progress"],
        "success_rate": share(statuses["completed"], finished),
        "blocked_rate": share(statuses["blocked"], finished),
        "failed_rate": share(statuses["failed"], finished),
        "n_replays": len(replays),
        "replay_match_rate": share(
            sum(1 for e in replays if e["details"].get("all_match")), len(replays)
        ),
        "cache_hit_rate": share(cache_hits, len(llm_plans)),
    }


def stage_latency(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean and 95th-percentile latency of each pipeline stage.

    Args:
        events: all events.

    Returns:
        One dict per stage with stage, count, mean_ms and p95_ms (None if no data).
    """
    rows = []
    for stage, event_types in STAGE_EVENTS.items():
        latencies = [
            e["latency_ms"]
            for e in events
            if e["event_type"] in event_types and e.get("latency_ms") is not None
        ]
        rows.append(
            {
                "stage": stage,
                "count": len(latencies),
                "mean_ms": round(float(np.mean(latencies)), 1) if latencies else None,
                "p95_ms": round(float(np.percentile(latencies, 95)), 1) if latencies else None,
            }
        )
    return rows


def capability_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    """How many times each capability was executed (completed or failed).

    Args:
        events: all events.

    Returns:
        capability -> count, most used first.
    """
    counts = Counter(
        e["details"].get("capability")
        for e in events
        if e["event_type"] in ("step_completed", "step_failed")
    )
    return dict(counts.most_common())


def guardrail_counts(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """How often each guardrail fired, by code and severity.

    Static issues are logged as validation_issue, runtime ones as guardrail_triggered.

    Args:
        events: all events.

    Returns:
        Rows of {code, name, severity, count}, most frequent first.
    """
    counts: Counter[tuple[str, str, str]] = Counter()
    for event in events:
        if event["event_type"] in ("validation_issue", "guardrail_triggered"):
            d = event["details"]
            counts[(d.get("code", "?"), d.get("name", "?"), d.get("severity", "?"))] += 1
    return [
        {"code": code, "name": name, "severity": severity, "count": count}
        for (code, name, severity), count in counts.most_common()
    ]


def recent_errors(events: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    """The most recent failures (planner errors, failed steps, failed or blocked runs).

    Args:
        events: all events.
        limit: maximum number of rows.

    Returns:
        Rows of {timestamp, run_id, event_type, error}, newest first.
    """
    error_types = ("plan_failed", "step_failed", "run_failed")
    rows = [
        {
            "timestamp": e["timestamp"],
            "run_id": e["run_id"],
            "event_type": e["event_type"],
            "error": e["details"].get("error", ""),
        }
        for e in events
        if e["event_type"] in error_types
    ]
    return list(reversed(rows))[:limit]
