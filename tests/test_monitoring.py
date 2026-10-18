"""Monitoring: the JSONL event log and the metrics behind the Monitoring page."""

from pathlib import Path
from typing import Any

import pytest

from quantlab.executor import StepRecord
from quantlab.guardrails import GUARDRAILS, ValidationReport, make_issue
from quantlab.lineage import LLMInfo, ParamChange, RunRecord
from quantlab.monitoring import (
    KPI_TARGETS,
    activity,
    approval_waits,
    cache_savings,
    capability_counts,
    cost_table,
    failure_point,
    grounded_narratives,
    guardrail_counts,
    latency_table,
    log_event,
    mode_counts,
    monitored_events,
    monitored_runs,
    planner_refusals,
    read_events,
    recent_failures,
    replay_results,
    review_outcomes,
    share,
    target_status,
    token_cost,
    unused_guardrails,
)
from quantlab.registry import REGISTRY
from quantlab.types import TokenUsage

PRICES = {"gpt-6-luna": {"input": 0.10, "output": 0.50}}


def record(
    run_id: str,
    created: str = "2026-10-01T10:00:00.000Z",
    *,
    status: str = "completed",
    mode: str = "live",
    model: str = "gpt-6-luna",
    prompt: str = "planner-v6",
    from_cache: bool = False,
    codes: tuple[str, ...] = (),
    runtime: tuple[str, ...] = (),
    step_status: tuple[str, ...] = (),
    **fields: Any,
) -> RunRecord:
    """A run record with only what a test needs (issues given by code)."""
    return RunRecord(
        run_id=run_id,
        created_at=created,
        status=status,
        request="test request",
        llm=LLMInfo(provider="openai", model=model, prompt_version=prompt, mode=mode,
                    from_cache=from_cache),
        validation=ValidationReport(issues=[make_issue(c, None, "msg", key=c) for c in codes]),
        runtime_issues=[make_issue(c, "s1", "msg", key=c) for c in runtime],
        steps=[StepRecord(step_id=f"s{i}", capability="load_series", library_calls="",
                          params={}, input_step_ids={}, input_hashes={}, status=s)
               for i, s in enumerate(step_status, start=1)],
        **fields,
    )  # fmt: skip


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


@pytest.mark.parametrize(
    ("metric", "value", "expected"),
    [
        # Met: exactly the target.
        ("replay_match_rate", 1.0, "met"),
        ("grounded_rate", 1.0, "met"),
        ("failed_share", 0.0, "met"),
        # Missed: anything else, however close.
        ("replay_match_rate", share(9999, 10000), "missed"),
        ("grounded_rate", share(9999, 10000), "missed"),
        ("failed_share", share(1, 10000), "missed"),
        # No data yet (no replays, no narratives, no runs): never shown as met.
        ("replay_match_rate", None, "no_data"),
        ("grounded_rate", None, "no_data"),
        ("failed_share", None, "no_data"),
    ],
)
def test_target_status(metric: str, value: float | None, expected: str) -> None:
    assert target_status(metric, value) == expected


def test_targets_and_exact_rates() -> None:
    assert KPI_TARGETS == {"replay_match_rate": 1.0, "grounded_rate": 1.0, "failed_share": 0.0}
    # No runs: the failed share has no data, so the Failed card is not green.
    assert target_status("failed_share", activity([])["failed_share"]) == "no_data"
    # Rates are not rounded, so a near miss cannot round up to the target.
    assert share(99999, 100000) != 1.0 and share(1, 100000) != 0.0
    assert (share(7, 7), share(0, 7), share(1, 0)) == (1.0, 0.0, None)


def test_token_cost() -> None:
    assert token_cost("gpt-6-luna", 1_000_000, 1_000_000, PRICES) == pytest.approx(0.60)
    # 1,500 input tokens at $0.10/M plus 400 output tokens at $0.50/M.
    assert token_cost("gpt-6-luna", 1500, 400, PRICES) == pytest.approx(0.00035)
    assert token_cost(None, 0, 0, PRICES) == 0.0  # no call, no cost
    assert token_cost("gpt-6-luna", None, 400, PRICES) is None  # not reported: unknown
    assert token_cost("unpriced-model", 10, 10, PRICES) is None


def test_monitored_runs_are_decided_runs_from_the_analysis_page() -> None:
    runs = [
        record("completed", interactive=True),
        record("blocked", status="blocked", interactive=True),  # never reached approval
        record("failed", status="failed", interactive=True),
        record("rejected", status="rejected", interactive=True),
        # Left out: undecided (pending or abandoned), headless, and of unknown origin.
        record("ready", status="ready", interactive=True),
        record("planned", status="planned", interactive=True),
        record("headless", interactive=False),
        record("old", interactive=None),
    ]
    assert [r.run_id for r in monitored_runs(runs)] == [
        "completed", "blocked", "failed", "rejected"]  # fmt: skip


def test_monitored_events_keep_replays_from_every_source() -> None:
    counted = [record("app", interactive=True)]
    events = [
        {"run_id": "app", "event_type": "validation_issue", "details": {"code": "V10"}},
        {"run_id": "cli", "event_type": "validation_issue", "details": {"code": "V08"}},
        {"run_id": "cli", "event_type": "replay_completed", "details": {"all_match": True}},
        {"run_id": "app", "event_type": "replay_completed", "details": {"all_match": False}},
    ]
    kept = monitored_events(events, counted)
    assert kept == [events[0], events[2], events[3]]  # the headless run's V08 is dropped
    assert replay_results(kept) == {"replays": 2, "replay_match_rate": 0.5}


def test_activity_counts_and_shares_add_up() -> None:
    runs = [
        record("a"),
        record("b", status="blocked"),
        record("c", status="blocked"),
        record("d", status="failed"),
        record("e", status="rejected"),
        record("f"),
        record("g"),
        record("h"),
    ]
    counts = activity(runs)
    assert counts == {
        "total": 8, "completed": 4, "blocked": 2, "failed": 1, "rejected": 1,
        "completed_share": 0.5, "blocked_share": 0.25, "failed_share": 0.125,
        "rejected_share": 0.125,
    }  # fmt: skip
    assert sum(v for k, v in counts.items() if k.endswith("_share")) == 1.0
    empty = activity([])
    assert empty["total"] == 0 and empty["completed_share"] is None


def test_mode_counts_add_up_to_the_runs() -> None:
    runs = [record("a"), record("b"), record("c", mode="cached"), record("d", mode="live")]
    counts = mode_counts(runs)
    assert counts == {"live": 3, "cached": 1, "mock": 0}  # a mode without runs is 0, listed
    assert sum(counts.values()) == len(runs)
    no_llm = record("e").model_copy(update={"llm": None})
    assert mode_counts([*runs, no_llm]) == {"live": 3, "cached": 1, "mock": 0, "unknown": 1}
    assert mode_counts([]) == {"live": 0, "cached": 0, "mock": 0}


def test_review_outcomes_approval_rate_counts_decisions_only() -> None:
    shown = "2026-10-01T10:00:00.000Z"
    edit = ParamChange(step_id="s4", param="window", original=12, final=26, edited_at=shown)
    runs = [
        # Approved after 30 s, with an edit; approved after 90 s, unedited; one rejected.
        record("a", plan_shown_at=shown, decision="approved",
               decided_at="2026-10-01T10:00:30.000Z", param_diff=[edit]),
        record("b", plan_shown_at=shown, decision="approved",
               decided_at="2026-10-01T10:01:30.000Z"),
        record("c", status="rejected", plan_shown_at=shown, decision="rejected",
               decided_at="2026-10-01T10:00:10.000Z"),
        # Blocked plans and refusals have no decision: not in the approval rate.
        record("d", status="blocked", plan_shown_at=shown),
        record("e", status="blocked", codes=("P01",)),
    ]  # fmt: skip
    review = review_outcomes(runs)
    assert (review["approved"], review["rejected"]) == (2, 1)
    assert review["approval_rate"] == pytest.approx(2 / 3)  # approved / (approved + rejected)
    assert review["edit_rate"] == 0.5
    assert review["approval_median_s"] == 60.0
    assert review["approval_p95_s"] == pytest.approx(87.0)  # 30 + 0.95 * (90 - 30)
    assert approval_waits(runs) == [30.0, 90.0]
    empty = review_outcomes([record("e", status="blocked")])
    assert empty["approval_rate"] is None and empty["approval_median_s"] is None


def test_replays_narratives_and_refusals() -> None:
    events = [
        {"run_id": "a", "event_type": "replay_completed", "details": {"all_match": True}},
        {"run_id": "a", "event_type": "replay_completed", "details": {"all_match": False}},
        {"run_id": "a", "event_type": "run_completed", "details": {}},
    ]
    assert replay_results(events) == {"replays": 2, "replay_match_rate": 0.5}
    assert replay_results([])["replay_match_rate"] is None

    n01 = {"issues": [{"code": "N01"}]}
    runs = [
        record("a", narrative=n01),  # an LLM narrative withheld by N01: not grounded
        record("b", narrative={"issues": []}),
        record("c", narrative={"issues": []}),
        record("d", narrative={"text": "made before N01 existed"}),  # left out
        record("e"),  # no narrative
    ]
    assert grounded_narratives(runs) == {"narratives": 3, "grounded_rate": pytest.approx(2 / 3)}
    assert grounded_narratives([])["grounded_rate"] is None

    runs = [
        record("a", status="blocked", codes=("P01",)),
        record("b"),
        record("c", status="blocked", codes=("R01",)),
        record("d", mode="mock"),
    ]
    # A count, and its share of all runs.
    assert planner_refusals(runs) == {"refused": 1, "refused_share": 0.25}
    assert planner_refusals([]) == {"refused": 0, "refused_share": None}


def test_guardrail_counts_list_every_guardrail_most_frequent_first() -> None:
    events = [
        {"run_id": "r", "event_type": "validation_issue", "details": {"code": "V08"}},
        {"run_id": "r", "event_type": "validation_issue", "details": {"code": "V08"}},
        {"run_id": "r", "event_type": "guardrail_triggered", "details": {"code": "E03"}},
        {"run_id": "r", "event_type": "validation_issue", "details": {"code": "G99"}},  # old
        {"run_id": "r", "event_type": "step_completed", "details": {"code": "V01"}},
    ]
    rows = guardrail_counts(events)
    assert [r["code"] for r in rows] == ["V08", "E03"] + [
        code for code in GUARDRAILS if code not in ("V08", "E03")]  # fmt: skip
    assert rows[0] == {"code": "V08", "name": "FREQUENCY_MISMATCH", "action": "BLOCK", "count": 2}
    assert [r["count"] for r in rows[2:]] == [0] * (len(GUARDRAILS) - 2)
    fired = guardrail_counts(events, include_unused=False)
    assert [(r["code"], r["count"]) for r in fired] == [("V08", 2), ("E03", 1)]
    assert guardrail_counts([], include_unused=False) == []
    unused = unused_guardrails(events)
    assert [r["code"] for r in unused] == sorted(set(GUARDRAILS) - {"V08", "E03"})  # by code
    assert unused[0] == {"code": "E01", "name": "DATASET_HASH_MISMATCH", "action": "BLOCK"}


def test_capability_counts_list_every_capability() -> None:
    events = [
        {"run_id": "r", "event_type": "step_completed", "details": {"capability": "resample"}},
        {"run_id": "r", "event_type": "step_failed", "details": {"capability": "resample"}},
        {"run_id": "r", "event_type": "step_completed", "details": {"capability": "spread"}},
    ]
    rows = capability_counts(events)
    assert [(r["capability"], r["count"]) for r in rows[:2]] == [("resample", 2), ("spread", 1)]
    assert len(rows) == len(REGISTRY) and rows[0]["code"] == REGISTRY["resample"].code
    assert [r["capability"] for r in rows[2:]] == [
        name for name in REGISTRY if name not in ("resample", "spread")]  # fmt: skip


def usage(input_tokens: int | None, output_tokens: int | None, **saved: int) -> TokenUsage:
    return TokenUsage(model="gpt-6-luna", input_tokens=input_tokens,
                      output_tokens=output_tokens, **saved)  # fmt: skip


def test_cost_table_and_cache_savings() -> None:
    planned = {"planning": 10.0}
    runs = [
        # Planner 1,000 in / 200 out ($0.0002); narrator 3,000 in / 400 out ($0.0005).
        record("a", stage_timings_ms=planned,
               stage_tokens={"planning": usage(1000, 200), "narration": usage(3000, 400)}),
        # Planner 3,000 in / 600 out ($0.0006); blocked, so no narration: zeros.
        record("b", status="blocked", stage_timings_ms=planned,
               stage_tokens={"planning": usage(3000, 600)}),
        # A cache hit: no planner tokens, the original call's 2,000 / 500 saved ($0.00045).
        record("c", from_cache=True, mode="cached", stage_timings_ms=planned,
               stage_tokens={"planning": usage(0, 0, saved_input_tokens=2000,
                                                 saved_output_tokens=500),
                             "narration": usage(1000, 0)}),
        # Left out of cost: a mock run, and a record from before tokens per stage.
        record("d", mode="mock", stage_timings_ms=planned,
               stage_tokens={"planning": TokenUsage()}),
        record("e", stage_timings_ms=planned),
    ]  # fmt: skip
    rows = {row["stage"]: row for row in cost_table(runs, PRICES)}
    assert rows["planner"]["runs"] == 3
    assert rows["planner"]["input_mean"] == pytest.approx(4000 / 3)
    assert rows["planner"]["cost_mean"] == pytest.approx(0.0008 / 3)
    assert rows["narrator"]["output_mean"] == pytest.approx(400 / 3)
    assert rows["narrator"]["cost_total"] == pytest.approx(0.0005 + 0.0001)
    assert rows["total"]["cost_total"] == pytest.approx(0.0014)
    # p95 of the per-run totals 0.0007, 0.0006 and 0.0001, by linear interpolation.
    assert rows["total"]["cost_p95"] == pytest.approx(0.0006 + 0.9 * 0.0001)

    savings = cache_savings(runs, PRICES)
    assert (savings["planner_calls"], savings["cache_hits"]) == (4, 1)  # d is mock
    assert savings["hit_rate"] == 0.25
    assert (savings["saved_input_tokens"], savings["saved_output_tokens"]) == (2000, 500)
    assert savings["saved_cost"] == pytest.approx(0.00045)


def test_unknown_tokens_or_prices_are_left_out_of_the_statistics() -> None:
    runs = [
        record("a", stage_tokens={"planning": usage(None, None)}),  # the call failed
        record("b", model="other", stage_tokens={
            "planning": TokenUsage(model="other", input_tokens=100, output_tokens=10)}),
    ]  # fmt: skip
    rows = {row["stage"]: row for row in cost_table(runs, PRICES)}
    assert rows["planner"]["input_mean"] == 100  # only b's count is known
    assert rows["planner"]["cost_mean"] is None and rows["total"]["cost_total"] is None
    assert cost_table([], PRICES)[0]["cost_mean"] is None


def test_latency_table_follows_the_pipeline() -> None:
    shown = "2026-10-01T10:00:00.000Z"
    runs = [
        record("r1", stage_timings_ms={"planning": 100.0, "validation": 2.0}),
        record("r2", stage_timings_ms={"planning": 300.0, "validation": 4.0, "execution": 50.0},
               interactive=True, plan_shown_at=shown, decision="approved",
               decided_at="2026-10-01T10:00:02.500Z"),
    ]  # fmt: skip
    rows = latency_table(runs)
    # Machine stages only: the 2.5 s r2 took to be approved is not a latency stage.
    assert [r["stage"] for r in rows] == [
        "Request check", "Planner", "Validator", "Executor", "Narrator"]  # fmt: skip
    by_stage = {r["stage"]: r for r in rows}
    assert by_stage["Planner"] == {"stage": "Planner", "runs": 2, "mean_ms": 200.0,
                                   "p95_ms": pytest.approx(290.0), "recorded": True}  # fmt: skip
    assert approval_waits(runs) == [2.5]  # it stays in the Approval time card
    # The request check is not timed in the run record; a stage no run reached has no data.
    assert by_stage["Request check"] == {"stage": "Request check", "runs": 0, "mean_ms": None,
                                         "p95_ms": None, "recorded": False}  # fmt: skip
    assert by_stage["Narrator"]["runs"] == 0 and by_stage["Narrator"]["recorded"]


def test_failure_point_names_the_stage_and_code() -> None:
    assert failure_point(record("a", status="blocked", codes=("R01",))) == ("Request check", "R01")
    assert failure_point(record("b", status="blocked", codes=("P01",))) == ("Planner", "P01")
    blocked = record("c", status="blocked", codes=("V10", "V08"))
    assert failure_point(blocked) == ("Validator", "V08")
    runtime = record("d", status="blocked", runtime=("E02",), step_status=("blocked",))
    assert failure_point(runtime) == ("Executor", "E02")
    crashed = record("e", status="failed", step_status=("completed", "failed"))
    assert failure_point(crashed) == ("Executor", None)
    unapproved = record("f", status="blocked", codes=("V10",))  # headless, not approved
    assert failure_point(unapproved) == ("Approval", "V10")


def test_recent_failures_newest_first() -> None:
    runs = [
        record("r1", status="blocked", codes=("V08",), error="frequency mismatch"),
        record("r2", status="completed"),
        record("r3", status="failed", step_status=("failed",), error="boom"),
        record("r4", status="rejected"),  # a decision, not a failure
    ]
    rows = recent_failures(runs, limit=5)
    assert [(r["run_id"], r["stage"], r["code"], r["message"]) for r in rows] == [
        ("r3", "Executor", None, "boom"),
        ("r1", "Validator", "V08", "frequency mismatch"),
    ]
    assert len(recent_failures(runs, limit=1)) == 1
    # Newest by creation time, even within one second (run ids then differ only randomly).
    same_second = [
        record("20261001-100000-ffffff", "2026-10-01T10:00:00.100Z", status="failed"),
        record("20261001-100000-000000", "2026-10-01T10:00:00.900Z", status="failed"),
    ]
    assert recent_failures(same_second)[0]["run_id"] == "20261001-100000-000000"
