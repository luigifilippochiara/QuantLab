"""In-plan parameter editing through the engine: values only, re-validated, re-approved.

The UI only offers widgets for registered parameters, but every rule is enforced
here, in the engine, so a crafted call cannot bypass it.
"""

from typing import Any

import pytest

from quantlab.config import Settings
from quantlab.engine import ExecutionRefusedError, Session
from quantlab.lineage import Approval, load_run, replay
from quantlab.mock_scenarios import SCENARIOS_BY_ID

from conftest import RETURN_METHOD_APPROVAL, SCENARIO_5_APPROVALS


def start(settings: Settings, scenario_id: str = "buy_the_dip_tech") -> Session:
    """A mock-mode session for a scenario, with its plan created and validated."""
    scenario = SCENARIOS_BY_ID[scenario_id]
    session = Session.start(scenario.request, list(scenario.dataset_ids), "mock", settings)
    session.create_plan()
    return session


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
    session = start(settings)
    before = session.plan.model_copy(deep=True)
    report = session.apply_param_edits(edits)
    rejected = [i for i in report.blocks if i.message.startswith("Edit rejected")]
    assert rejected and all(i.code == "V03" for i in rejected)
    assert session.plan == before  # nothing was applied
    assert session.status == "blocked" and not session.can_execute()[0]
    # A valid edit afterwards clears the block again.
    assert not session.apply_param_edits({"s4": {"min_periods": 104}}).is_blocked


def test_out_of_range_value_is_blocked_by_the_validator(settings: Settings) -> None:
    """The bounded widgets should prevent this; the validator is the backstop."""
    session = start(settings)
    report = session.apply_param_edits({"s4": {"quantile": 1.5}})
    assert report.is_blocked and "V03" in report.codes()
    assert session.plan.steps[3].params["quantile"] == 1.5  # applied, then judged
    with pytest.raises(ExecutionRefusedError):
        session.execute()


# --- re-validation through the same validator ------------------------------------


def test_switching_to_full_sample_raises_v09_and_switching_back_clears_it(
    settings: Settings,
) -> None:
    session = start(settings)
    report = session.apply_param_edits({"s4": {"method": "full_sample"}})
    assert "V09" in report.codes()
    report = session.apply_param_edits({"s4": {"method": "expanding"}})
    assert "V09" not in report.codes()


def test_weekly_to_daily_is_re_validated_and_runs_after_re_approval(settings: Settings) -> None:
    # The minimum history keeps its value, now counted in trading days instead of
    # weeks (the registry describes it in periods of the input's frequency).
    session = start(settings)
    report = session.apply_param_edits({"s2": {"freq": "D"}})
    assert report.warnings == [] and not report.is_blocked
    assert [i.issue_id for i in report.approvals_required] == SCENARIO_5_APPROVALS
    assert report.dry_run["s4"]["frequency"] == "D"
    session.approve(SCENARIO_5_APPROVALS)
    assert session.execute().status == "completed"
    flags = session.execution.artifacts["s4"].meta
    assert flags.frequency == "D"
    assert "(at least 52 past values)" in flags.notes[0]  # 52 trading days


# --- approvals belong to one plan version ----------------------------------------


def test_any_edit_resets_approvals_and_approvals_record_the_plan_hash(settings: Settings) -> None:
    session = start(settings)
    first_version = session.plan_hash
    session.approve(SCENARIO_5_APPROVALS)
    assert {a.plan_hash for a in session.approvals} == {first_version}
    session.apply_param_edits({"s4": {"min_periods": 104}})
    assert session.plan_hash != first_version
    assert session.approvals == []
    assert [i.issue_id for i in session.unresolved_approvals()] == SCENARIO_5_APPROVALS


def test_an_approval_for_another_plan_version_does_not_count(settings: Settings) -> None:
    session = start(settings)
    stale = Approval(issue_id=RETURN_METHOD_APPROVAL, code="V10", step_id="s3",
                     message="m", approved_at="t", plan_hash="another-version")  # fmt: skip
    session.approvals.append(stale)
    assert [i.issue_id for i in session.unresolved_approvals()] == SCENARIO_5_APPROVALS
    with pytest.raises(ExecutionRefusedError):
        session.execute()


def test_reset_to_the_planners_proposal(settings: Settings) -> None:
    session = start(settings)
    proposal = session.plan.model_copy(deep=True)
    session.apply_param_edits({"s2": {"freq": "D"}, "s4": {"min_periods": 60}})
    session.reset_to_proposal()
    assert session.plan.steps == proposal.steps
    assert session.build_record().param_diff == []
    assert session.report.codes() == ["V10"]  # the return-method approvals again


# --- lineage of edits ------------------------------------------------------------


def test_edited_run_records_the_diff_traces_it_and_replays_identically(
    settings: Settings,
) -> None:
    session = start(settings)
    session.apply_param_edits({"s4": {"min_periods": 104}})
    session.approve(SCENARIO_5_APPROVALS)
    session.execute()
    session.narrate()
    session.save_record()

    record = load_run(session.run_id, settings.runs_dir)
    # The planner left the minimum history to the registry's default (52), which is
    # what the diff shows as proposed.
    assert "min_periods" not in record.plan_original["steps"][3]["params"]
    assert record.plan_final["steps"][3]["params"]["min_periods"] == 104  # as run
    (change,) = record.param_diff
    assert (change.step_id, change.param, change.original, change.final) == (
        "s4", "min_periods", 52, 104)  # fmt: skip
    assert change.edited_at
    assert record.approvals[0].plan_hash == record.plan_hash

    chain = {step.step_id: step for step in session.trace("s6").chain}
    assert chain["s4"].edited == {"min_periods": 52}  # user-edited, planner proposed 52
    assert chain["s3"].edited == {}  # planner-proposed values only

    report = replay(session.run_id, settings)  # replays plan_final, without the LLM
    assert report.all_match, report.message
