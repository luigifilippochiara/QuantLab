"""Monitoring: an append-only JSONL event log, and the metrics behind the Monitoring page.

Role in the architecture: every stage of every run (and every replay) appends
one JSON object per line to runs/events.jsonl. Nothing is ever rewritten, so
the log is a simple audit trail and is trivial to ship to a real log platform
later.

The Monitoring page counts only decided runs started on the Analysis page
(monitored_runs); replays count from every source (monitored_events). It reads
two sources, and the functions below turn them into its numbers (the page only
lays them out):
- run records (runs/<run_id>/manifest.json), one per run in its final state:
  activity (planner refusals included), human review, narratives, warnings, tokens and cost,
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

from quantlab.config import LLM_MODES
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
    "grounded_rate": 1.0,  # every narrative must pass the grounding check (N01)
    # No failed run (not a block: an error to investigate). Checked as a share, so a
    # page with no runs has no data rather than meeting the target.
    "failed_share": 0.0,
}
# Statuses of a run whose plan reached the approval step with no decision yet
# (pending, or abandoned): left out of the Monitoring page. A blocked plan never
# reaches approval and keeps the status "blocked".
UNDECIDED = ("planned", "ready")


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
# What the page counts
# ---------------------------------------------------------------------------


def monitored_runs(records: list["RunRecord"]) -> list["RunRecord"]:
    """The runs the Monitoring page counts: started on the Analysis page, and decided.

    Left out: headless runs (CLI, notebooks, demo: interactive False), records made
    before the origin was recorded (interactive None: origin unknown), and runs
    whose plan awaits a decision or was abandoned (UNDECIDED).

    Args:
        records: run records.

    Returns:
        The counted records, in their original order.
    """
    return [r for r in records if r.interactive is True and r.status not in UNDECIDED]


def monitored_events(events: list[dict[str, Any]], runs: list["RunRecord"]) -> list[dict[str, Any]]:
    """The events of the counted runs, plus every replay, whatever run it replayed.

    Replays are usually started from Run history or the CLI rather than the Analysis
    page, so the replay match rate counts them from every source.

    Args:
        events: monitoring events.
        runs: the counted runs (monitored_runs).

    Returns:
        The kept events, in their original order.
    """
    run_ids = {r.run_id for r in runs}
    return [e for e in events if e["run_id"] in run_ids or e["event_type"] == "replay_completed"]


# ---------------------------------------------------------------------------
# Activity (run records)
# ---------------------------------------------------------------------------


def is_planner_refusal(record: "RunRecord") -> bool:
    """Whether the planner declined the request (P01): a blocked run, but not by a
    deterministic guardrail."""
    return record.status == "blocked" and bool(
        record.validation and "P01" in record.validation.codes()
    )


# The outcomes of the Activity section, in order: mutually exclusive, so over the
# counted runs their shares add up to 1.
OUTCOMES = ("completed", "blocked", "planner_refused", "failed", "rejected")


def whole_percentages(counts: list[int]) -> list[int] | None:
    """Each count's share as a whole percentage, the percentages adding up to exactly 100.

    Rounding each share on its own can give 99 or 101 (2/6 and four times 1/6 show
    33 + 4 x 17 = 101). The largest-remainder method rounds every share down, then
    gives the missing points, one each, to the shares that lost the most (the earliest
    on a tie).

    Args:
        counts: the counts of outcomes that make up a whole.

    Returns:
        One whole percentage per count, in the same order; None when the counts are all 0.
    """
    total = sum(counts)
    if total == 0:
        return None
    exact = [100 * count / total for count in counts]
    floors = [int(value) for value in exact]
    by_remainder = sorted(range(len(counts)), key=lambda i: floors[i] - exact[i])
    for i in by_remainder[: 100 - sum(floors)]:
        floors[i] += 1
    return floors


def activity(records: list["RunRecord"]) -> dict[str, Any]:
    """How many runs ended in each outcome, and each one's share of all runs.

    The outcomes do not overlap: "blocked" is a run stopped by a deterministic guardrail
    (the request check, the validator or the executor), "planner_refused" a run whose
    plan the planner declined (P01), and "rejected" a plan the reviewer turned down.

    Args:
        records: run records.

    Returns:
        Dict with total and a count per outcome in OUTCOMES, each outcome's share of
        total as "<outcome>_share" (None without runs), and its whole percentage for
        display as "<outcome>_percent" (see whole_percentages; None without runs). Over
        monitored_runs the counts add up to total, the shares to 1 and the
        percentages to 100.
    """
    refused = sum(1 for r in records if is_planner_refusal(r))
    statuses = Counter(r.status for r in records)
    counts = {
        "total": len(records),
        "completed": statuses["completed"],
        "blocked": statuses["blocked"] - refused,
        "planner_refused": refused,
        "failed": statuses["failed"],
        "rejected": statuses["rejected"],
    }
    shares = {f"{outcome}_share": share(counts[outcome], len(records)) for outcome in OUTCOMES}
    # For display: whole percentages of total, like the shares. Runs still awaiting a
    # decision (UNDECIDED; none in monitored_runs) are a last part that is not shown, so
    # the five outcomes add up to 100 exactly when they cover every run.
    parts = [counts[outcome] for outcome in OUTCOMES]
    rounded = whole_percentages([*parts, len(records) - sum(parts)])
    percents = {
        f"{outcome}_percent": None if rounded is None else rounded[i]
        for i, outcome in enumerate(OUTCOMES)
    }
    return counts | shares | percents


# The guardrails whose action is WARN, read from the guardrail table (never listed by
# hand), e.g. E03 MISSING_DATA or N01 UNGROUNDED_NARRATIVE.
WARN_CODES = frozenset(code for code, g in GUARDRAILS.items() if g.action == Action.WARN)


def completed_with_warnings(records: list["RunRecord"]) -> dict[str, Any]:
    """How many completed runs raised at least one WARN guardrail, and their share.

    Args:
        records: run records.

    Returns:
        Dict with completed, with_warnings and with_warnings_share (of completed runs;
        None when no run completed).
    """
    completed = [r for r in records if r.status == "completed"]

    def warned(record: "RunRecord") -> bool:
        issues = [*record.warnings, *record.runtime_issues]
        issues += record.validation.issues if record.validation else []
        return any(issue.code in WARN_CODES for issue in issues)

    with_warnings = sum(1 for r in completed if warned(r))
    return {
        "completed": len(completed),
        "with_warnings": with_warnings,
        "with_warnings_share": share(with_warnings, len(completed)),
    }


def mode_counts(records: list["RunRecord"]) -> dict[str, int]:
    """How many runs used each LLM mode, read from the run record (llm.mode).

    The mode is the one the run was started in: a cached-mode run counts as cached
    whether or not its plan came from the cache (the cache card counts the hits).

    Args:
        records: run records.

    Returns:
        live, cached and mock -> count, every mode listed (0 without runs), adding up
        to len(records); plus "unknown" only if some record names no mode (never one
        the page counts: every run saved by the engine records its LLM mode).
    """
    counts = Counter(r.llm.mode if r.llm else None for r in records)
    by_mode = {mode: counts[mode] for mode in LLM_MODES}
    unknown = len(records) - sum(by_mode.values())
    return by_mode | ({"unknown": unknown} if unknown else {})


# ---------------------------------------------------------------------------
# Quality and governance (run records, and replay events)
# ---------------------------------------------------------------------------


def approval_waits(records: list["RunRecord"]) -> list[float]:
    """Seconds from a plan being shown to its approval, for every approved run given.

    Older records lack the timestamps and are left out.

    Args:
        records: run records (on the page, monitored_runs: approved by a person).

    Returns:
        One wait per approved run, in seconds.
    """
    return [
        (parse_time(r.decided_at) - parse_time(r.plan_shown_at)).total_seconds()
        for r in records
        if r.decision == "approved" and r.plan_shown_at and r.decided_at
    ]


def review_outcomes(records: list["RunRecord"]) -> dict[str, Any]:
    """What reviewers decided about the plans they were shown.

    Args:
        records: run records (on the page, monitored_runs: decided on the Analysis page).

    Returns:
        Dict with approved, rejected, approval_rate (approved / (approved +
        rejected)), edited (approved plans whose parameters the reviewer changed),
        edit_rate (edited / approved), approval_median_s and approval_p95_s (seconds
        from plan shown to approved).
    """
    approved = [r for r in records if r.decision == "approved"]
    rejected = [r for r in records if r.decision == "rejected"]
    # param_diff: parameters whose final value differs from the planner's proposal.
    edited = sum(1 for r in approved if r.param_diff)
    waits = approval_waits(records)
    median, p95 = (
        (float(np.median(waits)), float(np.percentile(waits, 95))) if waits else (None, None)
    )
    return {
        "approved": len(approved),
        "rejected": len(rejected),
        "approval_rate": share(len(approved), len(approved) + len(rejected)),
        "edited": edited,
        "edit_rate": share(edited, len(approved)),
        "approval_median_s": median,
        "approval_p95_s": p95,
    }


def replay_results(events: list[dict[str, Any]]) -> dict[str, Any]:
    """How many replays ran and the share that reproduced their run exactly.

    Args:
        events: monitoring events.

    Returns:
        Dict with replays, matching (replays whose every output hash matched) and
        replay_match_rate (matching / replays; None without replays).
    """
    replays = [e for e in events if e["event_type"] == "replay_completed"]
    matching = sum(1 for e in replays if e["details"].get("all_match"))
    return {
        "replays": len(replays),
        "matching": matching,
        "replay_match_rate": share(matching, len(replays)),
    }


def grounded_narratives(records: list["RunRecord"]) -> dict[str, Any]:
    """The share of runs with a narrative whose narrative passed the grounding check.

    A narrative passes when the check raised no N01 issue (an LLM narrative withheld
    by N01 fails, even though the checked template is shown in its place).

    Args:
        records: run records (narratives recorded before N01 existed are left out).

    Returns:
        Dict with narratives, grounded (narratives that raised no N01) and grounded_rate
        (grounded / narratives; None without narratives).
    """
    narratives = [r.narrative for r in records if r.narrative and "issues" in r.narrative]
    grounded = sum(1 for n in narratives if not any(i["code"] == "N01" for i in n["issues"]))
    return {
        "narratives": len(narratives),
        "grounded": grounded,
        "grounded_rate": share(grounded, len(narratives)),
    }


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


def unused_guardrails(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The guardrails that never fired in these events.

    Args:
        events: monitoring events.

    Returns:
        Rows of {code, name, action}, sorted by code.
    """
    rows = [
        {"code": row["code"], "name": row["name"], "action": row["action"]}
        for row in guardrail_counts(events)
        if not row["count"]
    ]
    return sorted(rows, key=lambda row: row["code"])


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


# The machine stages of the pipeline in order, with the stage_timings_ms key that
# times each. Human approval is not a machine stage: its time is in review_outcomes.
LATENCY_STAGES = {
    "Request check": None,  # not timed in the run record (a word and character count)
    "Planner": "planning",
    "Validator": "validation",
    "Executor": "execution",
    "Narrator": "narration",
}


def latency_table(records: list["RunRecord"]) -> list[dict[str, Any]]:
    """Mean and 95th-percentile latency of each machine stage, in pipeline order.

    Each stage counts the runs that reached it. The time a person took to approve a
    plan is not included (it is the Approval time card: review_outcomes).

    Args:
        records: run records.

    Returns:
        One row per stage of LATENCY_STAGES: {stage, runs, mean_ms, p95_ms, recorded}
        (mean and p95 None without data; recorded False for the request check, which
        the run record does not time).
    """
    rows = []
    for stage, key in LATENCY_STAGES.items():
        values = [r.stage_timings_ms[key] for r in records if key and key in r.stage_timings_ms]
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
