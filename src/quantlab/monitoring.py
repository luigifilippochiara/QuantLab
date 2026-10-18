"""Monitoring: an append-only JSONL event log, and the metrics behind the Monitoring page.

Role in the architecture: every stage of every run (and every replay) appends
one JSON object per line to runs/events.jsonl. Nothing is ever rewritten, so
the log is a simple audit trail and is trivial to ship to a real log platform
later.

The Monitoring page reads two sources, and the functions below turn them into
its numbers (the page only lays them out):
- run records (runs/<run_id>/manifest.json), one per run in its final state:
  activity, human review, narrative and planner outcomes, tokens and cost,
  latency and failures;
- the event log, for what can happen several times in a run or after it:
  guardrail firings (one raised by an edit and then fixed still counts),
  capability executions and replays.

The event types are listed in EVENT_TYPES below: the spec's list plus
"plan_validated" and "execution_finished", which carry the validation and
execution stage latencies, and "plan_rejected", the reviewer's rejection.
"""

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from quantlab.guardrails import ENFORCERS, GUARDRAILS, Action
from quantlab.registry import REGISTRY
from quantlab.types import TokenUsage

if TYPE_CHECKING:
    # For type hints only: lineage imports this module (to log replays), so a
    # runtime import of lineage here would be circular.
    from quantlab.lineage import RunRecord

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
    "plan_rejected",
    "replay_completed",
)

# The stages that call an LLM (keys of stage_tokens), and their names in the cost table.
COST_STAGES = {"planning": "planner", "narration": "narrator"}

# The Monitoring cards with a known correct value: metric -> target. A card is green
# when its value equals the target, orange when it does not, neutral without data
# (see target_status). Kept in one place so the targets are easy to review.
KPI_TARGETS = {
    "replay_match_rate": 1.0,  # every replay must reproduce its run exactly
    "n01_failure_rate": 0.0,  # no narrative should quote an unchecked number
    "failed": 0,  # a failed run (not a block) is an error to investigate
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
        details: extra fields (capability, issue code, action, error message...).

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
# Helpers
# ---------------------------------------------------------------------------


def target_status(metric: str, value: float | None) -> str:
    """Whether a metric meets its target in KPI_TARGETS.

    The comparison is exact: rates are unrounded (see share), so 9,999 matching
    replays out of 10,000 miss a 100% target. A metric without data is never "met".

    Args:
        metric: a key of KPI_TARGETS.
        value: the metric's value, or None when there is no data yet (e.g. no replays).

    Returns:
        "met", "missed" or "no_data".
    """
    if value is None:
        return "no_data"
    return "met" if value == KPI_TARGETS[metric] else "missed"


def share(count: int, total: int) -> float | None:
    """count / total, or None when there is nothing to divide by.

    Not rounded: a rounded 99,999 / 100,000 would read as exactly 1.0 and meet a 100%
    target it misses. The page formats rates for display.
    """
    return count / total if total else None


def mean_and_p95(values: list[float]) -> tuple[float | None, float | None]:
    """Mean and 95th percentile (linear interpolation), or (None, None) without values."""
    if not values:
        return None, None
    return float(np.mean(values)), float(np.percentile(values, 95))


def parse_time(timestamp: str) -> datetime:
    """Parse a UTC timestamp written by utc_now() or a run record, e.g. 2026-10-14T19:24:42.123Z."""
    return datetime.fromisoformat(timestamp)


def token_cost(
    model: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
    prices: dict[str, dict[str, float]],
) -> float | None:
    """Cost in US dollars of a number of tokens at a model's price.

    Args:
        model: the model id (the key into `prices`).
        input_tokens: input tokens (None: not reported).
        output_tokens: output tokens (None: not reported).
        prices: model -> {"input": $ per million tokens, "output": $ per million tokens}
            (config.LLM_PRICES_PER_MILLION_TOKENS).

    Returns:
        The cost; 0.0 for no tokens; None if a count is missing or the model has no price.
    """
    if input_tokens is None or output_tokens is None:
        return None
    if input_tokens == 0 and output_tokens == 0:
        return 0.0
    price = prices.get(model or "")
    if price is None:
        return None
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def run_order(record: "RunRecord") -> tuple[datetime, str]:
    """Sort key, oldest first: the creation time (to the millisecond), then the run id.

    The run id alone is not enough: two runs created in the same second differ only
    in their random suffix.
    """
    return parse_time(record.created_at), record.run_id


# ---------------------------------------------------------------------------
# Activity (run records)
# ---------------------------------------------------------------------------


def activity(records: list["RunRecord"]) -> dict[str, Any]:
    """How many runs ended in each state, and each one's share of all runs.

    Args:
        records: run records.

    Returns:
        Dict with total, completed, blocked, failed, rejected (by the reviewer) and
        awaiting (a plan shown with no decision yet: status ready), and
        completed_share, blocked_share, failed_share and rejected_share (of total;
        None without runs).
    """
    statuses = Counter(r.status for r in records)
    counts = {
        "total": len(records),
        "completed": statuses["completed"],
        "blocked": statuses["blocked"],
        "failed": statuses["failed"],
        "rejected": statuses["rejected"],
        "awaiting": statuses["ready"] + statuses["planned"],
    }
    shares = {
        f"{outcome}_share": share(counts[outcome], len(records))
        for outcome in ("completed", "blocked", "failed", "rejected")
    }
    return counts | shares


# ---------------------------------------------------------------------------
# Quality and governance (run records, and replay events)
# ---------------------------------------------------------------------------


def approval_waits(records: list["RunRecord"]) -> list[float]:
    """Seconds from a plan being shown to its approval, for plans approved on the Analysis page.

    Headless runs approve automatically (in milliseconds), and older records lack the
    timestamps: both are left out.

    Args:
        records: run records.

    Returns:
        One wait per interactive, approved run, in seconds.
    """
    return [
        (parse_time(r.decided_at) - parse_time(r.plan_shown_at)).total_seconds()
        for r in records
        if r.interactive and r.plan_shown_at and r.decision == "approved" and r.decided_at
    ]


def review_outcomes(records: list["RunRecord"]) -> dict[str, Any]:
    """What reviewers decided about the plans shown to them on the Analysis page.

    Only interactive runs with a plan shown count: headless runs (CLI, notebooks,
    demo) are decided by their caller, not reviewed by a person, and older records
    lack the fields. A plan shown but neither approved (executed) nor rejected is
    abandoned, including one still awaiting a decision.

    Args:
        records: run records.

    Returns:
        Dict with shown, approved, rejected, abandoned, approval_rate (approved /
        shown), edit_rate (approved plans whose parameters the reviewer changed /
        approved), approval_median_s and approval_p95_s (seconds from plan shown to
        approved), and headless (headless runs with a plan, left out).
    """
    shown = [r for r in records if r.interactive and r.plan_shown_at]
    approved = [r for r in shown if r.decision == "approved"]
    rejected = [r for r in shown if r.decision == "rejected"]
    waits = approval_waits(records)
    median, p95 = (
        (float(np.median(waits)), float(np.percentile(waits, 95))) if waits else (None, None)
    )
    return {
        "shown": len(shown),
        "approved": len(approved),
        "rejected": len(rejected),
        "abandoned": len(shown) - len(approved) - len(rejected),
        "approval_rate": share(len(approved), len(shown)),
        # param_diff: parameters whose final value differs from the planner's proposal.
        "edit_rate": share(sum(1 for r in approved if r.param_diff), len(approved)),
        "approval_median_s": median,
        "approval_p95_s": p95,
        "headless": sum(1 for r in records if r.interactive is False and r.plan_shown_at),
    }


def replay_results(events: list[dict[str, Any]]) -> dict[str, Any]:
    """How many replays ran and the share that reproduced their run exactly.

    Args:
        events: monitoring events.

    Returns:
        Dict with replays and replay_match_rate (None without replays).
    """
    replays = [e for e in events if e["event_type"] == "replay_completed"]
    matching = sum(1 for e in replays if e["details"].get("all_match"))
    return {"replays": len(replays), "replay_match_rate": share(matching, len(replays))}


def narrative_failures(records: list["RunRecord"]) -> dict[str, Any]:
    """The share of narratives that failed the grounding check (N01).

    Args:
        records: run records (narratives recorded before N01 existed are left out).

    Returns:
        Dict with narratives and n01_failure_rate (None without narratives).
    """
    narratives = [r.narrative for r in records if r.narrative and "issues" in r.narrative]
    failed = sum(1 for n in narratives if any(i["code"] == "N01" for i in n["issues"]))
    return {"narratives": len(narratives), "n01_failure_rate": share(failed, len(narratives))}


def planner_refusals(records: list["RunRecord"]) -> dict[str, Any]:
    """The share of requests the planner was asked about that it refused (P01).

    Args:
        records: run records (a request refused before the planner, R01, is left out).

    Returns:
        Dict with planned (requests that reached the planner) and p01_refusal_rate.
    """
    codes = [r.validation.codes() if r.validation else [] for r in records if r.llm]
    planned = [c for c in codes if "R01" not in c]
    refused = sum(1 for c in planned if "P01" in c)
    return {"planned": len(planned), "p01_refusal_rate": share(refused, len(planned))}


# ---------------------------------------------------------------------------
# Guardrails and capabilities (events)
# ---------------------------------------------------------------------------


def guardrail_counts(
    events: list[dict[str, Any]], include_unused: bool = True
) -> list[dict[str, Any]]:
    """How often each guardrail fired: every guardrail in the table, zeros included.

    Request-check, planner-refusal and validator issues are logged as
    validation_issue; the executor's and the narrator check's as guardrail_triggered.
    A code that is no longer in the table (from an older version) is not counted.

    Args:
        events: monitoring events.
        include_unused: False to keep only the guardrails that fired (count above zero).

    Returns:
        Rows of {code, name, action, count}, most frequent first; guardrails with
        the same count keep the table's order, so the zeros come last in pipeline order.
    """
    counts = Counter(
        e["details"].get("code")
        for e in events
        if e["event_type"] in ("validation_issue", "guardrail_triggered")
    )
    rows = [
        {"code": code, "name": g.name, "action": g.action.value, "count": counts[code]}
        for code, g in GUARDRAILS.items()
    ]
    if not include_unused:
        rows = [row for row in rows if row["count"]]
    return sorted(rows, key=lambda row: -row["count"])  # a stable sort keeps the ties in order


def capability_counts(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """How many times each capability was executed: every registered one, zeros included.

    Args:
        events: monitoring events (step_completed and step_failed count).

    Returns:
        Rows of {capability, code, count}, most used first; ties keep the registry's order.
    """
    counts = Counter(
        e["details"].get("capability")
        for e in events
        if e["event_type"] in ("step_completed", "step_failed")
    )
    rows = [
        {"capability": name, "code": spec.code, "count": counts[name]}
        for name, spec in REGISTRY.items()
    ]
    return sorted(rows, key=lambda row: -row["count"])


# ---------------------------------------------------------------------------
# Cost (run records)
# ---------------------------------------------------------------------------


def costed_runs(records: list["RunRecord"]) -> list["RunRecord"]:
    """The runs that count towards cost: not mock, with tokens recorded per stage."""
    return [r for r in records if r.stage_tokens is not None and r.llm and r.llm.mode != "mock"]


def _stage_values(record: "RunRecord", prices: dict[str, dict[str, float]]) -> dict[str, dict]:
    """One run's input tokens, output tokens and cost per cost stage, and in total.

    A stage that did not run used nothing (zeros). A value that is unknown (tokens
    not reported, or a model without a price) is None, and so is the total built on it.
    """
    values = {}
    for stage, name in COST_STAGES.items():
        usage = record.stage_tokens.get(stage, TokenUsage())
        values[name] = {
            "input": usage.input_tokens,
            "output": usage.output_tokens,
            "cost": token_cost(usage.model, usage.input_tokens, usage.output_tokens, prices),
        }
    values["total"] = {
        measure: None
        if any(v[measure] is None for v in values.values())
        else sum(v[measure] for v in values.values())
        for measure in ("input", "output", "cost")
    }
    return values


def cost_table(
    records: list["RunRecord"], prices: dict[str, dict[str, float]]
) -> list[dict[str, Any]]:
    """Tokens and cost per run, by stage (planner, narrator, total): mean and 95th percentile.

    Args:
        records: run records; only costed_runs(records) count (mock runs are left out).
        prices: model -> price per million tokens (config.LLM_PRICES_PER_MILLION_TOKENS).

    Returns:
        Rows of {stage, runs, input_mean, input_p95, output_mean, output_p95,
        cost_mean, cost_p95, cost_total} (None where no value is known).
    """
    per_run = [_stage_values(r, prices) for r in costed_runs(records)]
    rows = []
    for stage in [*COST_STAGES.values(), "total"]:
        row: dict[str, Any] = {"stage": stage, "runs": len(per_run)}
        for measure in ("input", "output", "cost"):
            known = [v[stage][measure] for v in per_run if v[stage][measure] is not None]
            row[f"{measure}_mean"], row[f"{measure}_p95"] = mean_and_p95(known)
        costs = [v[stage]["cost"] for v in per_run if v[stage]["cost"] is not None]
        row["cost_total"] = sum(costs) if costs else None
        rows.append(row)
    return rows


def cache_savings(
    records: list["RunRecord"], prices: dict[str, dict[str, float]]
) -> dict[str, Any]:
    """The plan-cache hit rate, and the tokens and dollars the hits saved.

    Args:
        records: run records; runs that called (or would have called) the planner
            in live or cached mode count; mock runs and R01 refusals do not.
        prices: model -> price per million tokens.

    Returns:
        Dict with planner_calls, cache_hits, hit_rate, saved_input_tokens,
        saved_output_tokens and saved_cost (US dollars; hits recorded before tokens
        were recorded per stage add nothing).
    """
    calls = [
        r for r in records if r.llm and r.llm.mode != "mock" and "planning" in r.stage_timings_ms
    ]
    hits = [r for r in calls if r.llm.from_cache]
    saved = [
        r.stage_tokens["planning"] for r in hits if r.stage_tokens and "planning" in r.stage_tokens
    ]
    costs = [
        token_cost(u.model, u.saved_input_tokens, u.saved_output_tokens, prices) for u in saved
    ]
    return {
        "planner_calls": len(calls),
        "cache_hits": len(hits),
        "hit_rate": share(len(hits), len(calls)),
        "saved_input_tokens": sum(u.saved_input_tokens for u in saved),
        "saved_output_tokens": sum(u.saved_output_tokens for u in saved),
        "saved_cost": sum(c for c in costs if c is not None),
    }


# ---------------------------------------------------------------------------
# Latency and failures (run records)
# ---------------------------------------------------------------------------


# The pipeline stages in order, with the stage_timings_ms key that times each.
LATENCY_STAGES = {
    "Request check": None,  # not timed in the run record (a word and character count)
    "Planner": "planning",
    "Validator": "validation",
    "Approval": "approval",  # not a stage_timings_ms key: see approval_waits
    "Executor": "execution",
    "Narrator": "narration",
}


def latency_table(records: list["RunRecord"]) -> list[dict[str, Any]]:
    """Mean and 95th-percentile latency of each pipeline stage, in pipeline order.

    Each stage counts the runs that reached it; approval counts plans approved on the
    Analysis page (approval_waits), the time a person took to decide.

    Args:
        records: run records.

    Returns:
        One row per stage of LATENCY_STAGES: {stage, runs, mean_ms, p95_ms, recorded}
        (mean and p95 None without data; recorded False for the request check, which
        the run record does not time).
    """
    rows = []
    for stage, key in LATENCY_STAGES.items():
        if key is None:
            values = []
        elif key == "approval":
            values = [seconds * 1000 for seconds in approval_waits(records)]
        else:
            values = [r.stage_timings_ms[key] for r in records if key in r.stage_timings_ms]
        mean, p95 = mean_and_p95(values)
        rows.append(
            {
                "stage": stage,
                "runs": len(values),
                "mean_ms": mean,
                "p95_ms": p95,
                "recorded": key is not None,
            }
        )
    return rows


def failure_point(record: "RunRecord") -> tuple[str, str | None]:
    """Where a blocked or failed run stopped: the stage, and the guardrail code if one fired.

    Args:
        record: a blocked or failed run record.

    Returns:
        (stage, code): the stage enforcing the first BLOCK issue (request check,
        planner, validator or executor), "Executor" for a step that failed without
        one, "Approval" for a plan left unapproved in a headless run, or "Unknown".
    """
    issues = record.validation.issues if record.validation else []
    blocks = [i for i in issues + record.runtime_issues if i.action == Action.BLOCK]
    if blocks:
        code = blocks[0].code
        return ENFORCERS.get(code[0], "Unknown"), code
    if any(step.status != "completed" for step in record.steps):
        return ENFORCERS["E"], None
    approvals = [i for i in issues if i.action == Action.APPROVE]
    if approvals and not record.steps:
        return "Approval", approvals[0].code
    return "Unknown", None


def recent_failures(records: list["RunRecord"], limit: int = 10) -> list[dict[str, Any]]:
    """The most recent blocked or failed runs, with where they stopped and why.

    Args:
        records: run records.
        limit: maximum number of rows.

    Returns:
        Rows of {time, run_id, status, stage, code, message}, newest first.
    """
    failed = sorted(
        (r for r in records if r.status in ("blocked", "failed")), key=run_order, reverse=True
    )[:limit]
    rows = []
    for r in failed:
        stage, code = failure_point(r)
        rows.append(
            {
                "time": r.created_at,
                "run_id": r.run_id,
                "status": r.status,
                "stage": stage,
                "code": code,
                "message": r.error or "",
            }
        )
    return rows
