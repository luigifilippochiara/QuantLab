"""In-plan parameter editing through the graph: values only, re-validated, re-approved.

The UI only offers widgets for registered parameters, but every rule is enforced
by the graph (check_edits, then the validator node), so a crafted answer to the
approval interrupt cannot bypass it. The runs here are reviewed by a person
(interactive), as on the Analysis page: a blocked plan waits for the next edit.
"""

from typing import Any

import pytest

from quantlab.config import Settings
from quantlab.graph import (
    current_plan_hash,
    refusal_reason,
    reset_edits,
    resume_run,
    run_graph,
    unresolved_approvals,
)
from quantlab.lineage import Approval, load_run, replay, trace
from quantlab.mock_scenarios import SCENARIOS_BY_ID
from quantlab.monitoring import read_events

from conftest import (
    RETURN_METHOD_APPROVAL,
    SCENARIO_5_APPROVALS,
    edit,
    execute,
    paused_ids,
    start_review,
)

S5 = "buy_the_dip_tech"

# --- the structure is locked -------------------------------------------------------


@pytest.mark.parametrize(
    "edits",
    [
        pytest.param({"s9": {"window": 3}}, id="add a step"),
        pytest.param({"s4": {"capability": "arima"}}, id="change the capability"),
        pytest.param({"s4": {"inputs": {"series": "s1"}}}, id="change an input reference"),
        pytest.param({"s4": {"id": "s1"}}, id="change a step id"),
        pytest.param({"s4": {"code": "import os"}}, id="invent a field"),
        pytest.param({"s4": ["window", 26]}, id="params not a mapping"),
        pytest.param([("s4", "window", 26)], id="edits not a mapping"),
    ],
)
def test_structural_edits_are_rejected_with_a_block(edits: Any, settings: Settings) -> None:
    graph, run = start_review(S5, settings)
    before = run.state["plan"].model_copy(deep=True)
    run = edit(graph, run, edits)
    report = run.state["report"]
    rejected = [i for i in report.blocks if i.message.startswith("Edit rejected")]
    assert rejected and all(i.code == "V03" for i in rejected)
    assert run.state["plan"] == before  # nothing was applied
    assert run.state["status"] == "blocked" and refusal_reason(run.state)
    # A valid edit afterwards clears the block again.
    assert not edit(graph, run, {"s4": {"min_periods": 104}}).state["report"].is_blocked


def test_out_of_range_value_is_blocked_by_the_validator(settings: Settings) -> None:
    """The bounded widgets should prevent this; the validator is the backstop."""
    graph, run = start_review(S5, settings)
    run = edit(graph, run, {"s4": {"quantile": 1.5}})
    report = run.state["report"]
    assert report.is_blocked and "V03" in report.codes()
    assert run.state["plan"].steps[3].params["quantile"] == 1.5  # applied, then judged
    refused = execute(graph, run, SCENARIO_5_APPROVALS)
    assert refused.paused_for["error"].startswith("The plan has blocking issues: V03")
    assert "steps" not in refused.state  # nothing ran
    # Fixing the value unblocks it again.
    assert not edit(graph, refused, {"s4": {"quantile": 0.1}}).state["report"].is_blocked


# --- re-validation through the same validator ------------------------------------


def test_switching_to_full_sample_raises_v09_and_switching_back_clears_it(
    settings: Settings,
) -> None:
    graph, run = start_review(S5, settings)
    run = edit(graph, run, {"s4": {"method": "full_sample"}})
    assert "V09" in run.state["report"].codes()
    run = edit(graph, run, {"s4": {"method": "expanding"}})
    assert "V09" not in run.state["report"].codes()


def test_weekly_to_daily_is_re_validated_and_runs_after_re_approval(settings: Settings) -> None:
    # The minimum history keeps its value, now counted in trading days instead of
    # weeks (the registry describes it in periods of the input's frequency).
    graph, run = start_review(S5, settings)
    run = edit(graph, run, {"s2": {"freq": "D"}})
    report = run.state["report"]
    assert report.warnings == [] and not report.is_blocked
    assert [i.issue_id for i in report.approvals_required] == SCENARIO_5_APPROVALS
    assert report.dry_run["s4"]["frequency"] == "D"
    run = execute(graph, run, SCENARIO_5_APPROVALS)
    assert run.state["status"] == "completed"
    flags = run.state["artifacts"]["s4"].meta
    assert flags.frequency == "D"
    assert "(at least 52 past values)" in flags.notes[0]  # 52 trading days


# --- approvals belong to one plan version ----------------------------------------


def approvals_for(run: Any, plan_hash: str) -> list[Approval]:
    """Approvals of every open issue of a paused run, given for the plan version `plan_hash`."""
    return [
        Approval(
            issue_id=i.issue_id,
            code=i.code,
            step_id=i.step_id,
            message=i.message,
            approved_at="t",
            plan_hash=plan_hash,
        )
        for i in unresolved_approvals(run.state)
    ]


def test_any_edit_resets_approvals_and_approvals_record_the_plan_hash(settings: Settings) -> None:
    graph, run = start_review(S5, settings)
    first_version = current_plan_hash(run.state)
    # Through the interrupt, approvals are given with Execute; put them in the paused
    # run's checkpoint to show that an edit clears approvals of the version before.
    graph.update_state(run.config, {"approvals": approvals_for(run, first_version)})
    assert unresolved_approvals(graph.get_state(run.config).values) == []
    run = edit(graph, run, {"s4": {"min_periods": 104}})
    assert current_plan_hash(run.state) != first_version
    assert run.state["approvals"] == []
    assert paused_ids(run) == SCENARIO_5_APPROVALS
    run = execute(graph, run, SCENARIO_5_APPROVALS)
    assert {a.plan_hash for a in run.state["approvals"]} == {current_plan_hash(run.state)}


def test_an_approval_for_another_plan_version_does_not_count(settings: Settings) -> None:
    graph, run = start_review(S5, settings)
    stale = Approval(issue_id=RETURN_METHOD_APPROVAL, code="V10", step_id="s3",
                     message="m", approved_at="t", plan_hash="another-version")  # fmt: skip
    graph.update_state(run.config, {"approvals": [stale]})
    state = graph.get_state(run.config).values
    assert [i.issue_id for i in unresolved_approvals(state)] == SCENARIO_5_APPROVALS
    refused = execute(graph, run, [])
    assert refused.paused_for["error"] == "Approval required for: " + ", ".join(
        SCENARIO_5_APPROVALS
    )
    assert "steps" not in refused.state  # nothing ran


def test_reset_to_the_planners_proposal(settings: Settings) -> None:
    graph, run = start_review(S5, settings)
    proposal = run.state["plan"].model_copy(deep=True)
    run = edit(graph, run, {"s2": {"freq": "D"}, "s4": {"min_periods": 60}})
    run = edit(graph, run, reset_edits(run.state))
    assert run.state["plan"].steps == proposal.steps
    # The record saved for the plan awaiting review has no parameter change left.
    assert load_run(run.state["run_id"], settings.runs_dir).param_diff == []
    assert run.state["report"].codes() == ["V10"]  # the return-method approvals again


# --- lineage of edits ------------------------------------------------------------


def test_edited_run_records_the_diff_traces_it_and_replays_identically(
    settings: Settings,
) -> None:
    graph, run = start_review(S5, settings)
    run = edit(graph, run, {"s4": {"min_periods": 104}})
    run = execute(graph, run, SCENARIO_5_APPROVALS)

    record = load_run(run.state["run_id"], settings.runs_dir)
    # The planner left the minimum history to the registry's default (52), which is
    # what the diff shows as proposed.
    assert "min_periods" not in record.plan_original["steps"][3]["params"]
    assert record.plan_final["steps"][3]["params"]["min_periods"] == 104  # as run
    (change,) = record.param_diff
    assert (change.step_id, change.param, change.original, change.final) == (
        "s4", "min_periods", 52, 104)  # fmt: skip
    assert change.edited_at
    assert record.approvals[0].plan_hash == record.plan_hash

    chain = {step.step_id: step for step in trace(run.state["record"], "s6").chain}
    assert chain["s4"].edited == {"min_periods": 52}  # user-edited, planner proposed 52
    assert chain["s3"].edited == {}  # planner-proposed values only

    report = replay(run.state["run_id"], settings)  # replays plan_final, without the LLM
    assert report.all_match, report.message


# --- a plan rejected as a whole (V01, V02) is final -------------------------------


def never_called(*args: Any, **kwargs: Any) -> None:
    """Stands in for the executor: a blocked plan must never reach it."""
    raise AssertionError("the executor was called with a blocked plan")


def test_a_rejected_plan_cannot_be_edited_approved_rejected_or_executed(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("quantlab.graph.execute_plan", never_called)
    graph, run = start_review("prompt_injection_unknown_capability", settings)
    assert run.state["report"].is_rejected and run.state["status"] == "blocked"
    # The validator rejected the plan: the run went to the recorder and ended, so there
    # is no review to answer, even for a person reviewing.
    assert run.paused_for is None and graph.get_state(run.config).next == ()
    assert refusal_reason(run.state) == "The plan was rejected (V02): write a new request."
    events = read_events(settings.events_path)
    for decision in (
        {"action": "edit", "edits": {"s1": {"code": "print(1)"}}},
        {"action": "edit", "edits": reset_edits(run.state)},
        {"action": "approve", "issue_ids": []},
        {"action": "reject"},
    ):
        # An answer to a run that has ended changes nothing and logs nothing.
        assert resume_run(graph, run, decision).state == run.state
    assert read_events(settings.events_path) == events
    assert "steps" not in run.state and run.state.get("decision") is None


@pytest.mark.parametrize(
    "scenario_id", ["prompt_injection_unknown_capability", "spread_frequency_mismatch_block"]
)
def test_a_blocked_plan_never_reaches_the_executor(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, scenario_id: str
) -> None:
    # Rejected (V02) or fixable (V08): blocked by the validator, never executed.
    monkeypatch.setattr("quantlab.graph.execute_plan", never_called)
    graph, run = start_review(scenario_id, settings)
    assert run.state["report"].is_blocked
    if run.paused_for is not None:  # fixable: Execute, with every approval, is refused
        run = execute(graph, run, paused_ids(run))
        assert run.paused_for["error"].startswith("The plan has blocking issues")
    assert "steps" not in run.state
    # Headless (CLI --approve-all), a blocked plan ends the run before any review.
    scenario = SCENARIOS_BY_ID[scenario_id]
    state = run_graph(scenario.request, list(scenario.dataset_ids), "mock", True, settings)
    assert state["status"] == "blocked" and "steps" not in state


def test_a_v08_block_is_still_fixed_by_an_edit_and_runs(settings: Settings) -> None:
    graph, run = start_review("spread_frequency_mismatch_block", settings)
    report = run.state["report"]
    assert [i.code for i in report.blocks] == ["V08"] and not report.is_rejected
    # The editable fix: the 10-year's resample to daily, so both yields are daily.
    run = edit(graph, run, {"s2": {"freq": "D"}})
    report = run.state["report"]
    assert not report.is_blocked and [i.issue_id for i in report.approvals_required] == [
        "V09:s5"]  # fmt: skip
    assert execute(graph, run, ["V09:s5"]).state["status"] == "completed"


def test_scenario_4_loads_no_data_and_runs_no_step(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_loading(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a dataset was loaded")

    monkeypatch.setattr("quantlab.capabilities.load_dataset", no_loading)
    monkeypatch.setattr("quantlab.graph.execute_plan", never_called)
    scenario = SCENARIOS_BY_ID["prompt_injection_unknown_capability"]
    state = run_graph(scenario.request, list(scenario.dataset_ids), "mock", settings=settings)
    assert state["status"] == "blocked" and "steps" not in state
    record = load_run(state["run_id"], settings.runs_dir)
    assert record.steps == [] and record.datasets == [] and record.status == "blocked"
    # No execution or step event was logged for this run.
    events = [e for e in read_events(settings.events_path) if e["run_id"] == state["run_id"]]
    assert not [e for e in events if e["event_type"].startswith(("execution", "step_"))]
