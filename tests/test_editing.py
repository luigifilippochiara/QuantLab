"""In-plan parameter editing through the engine: values only, re-validated, re-approved.

The UI only offers widgets for registered parameters, but every rule is enforced
here, in the engine, so a crafted call cannot bypass it.
"""

import copy
from typing import Any

import pytest

from quantlab.config import Settings
from quantlab.engine import ExecutionRefusedError, Session
from quantlab.lineage import Approval, load_run, replay
from quantlab.mock_scenarios import SCENARIOS_BY_ID
from quantlab.narrator import build_facts_sheet, narrate

from conftest import PERIOD_PLAN, RETURN_METHOD_APPROVAL, canned_plan
from fakes import FakeAnthropicClient, text_response, tool_response


def start(settings: Settings, scenario_id: str = "volatility_flag_success") -> Session:
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
        pytest.param({"s4": {"capability": "sarima"}}, id="change the capability"),
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
    assert rejected and all(i.code == "G03" for i in rejected)
    assert session.plan == before  # nothing was applied
    assert session.status == "blocked" and not session.can_execute()[0]
    # A valid edit afterwards clears the block again.
    assert not session.apply_param_edits({"s4": {"window": 26}}).is_blocked


def test_out_of_range_value_is_blocked_by_the_validator(settings: Settings) -> None:
    """The bounded widgets should prevent this; the validator is the backstop."""
    session = start(settings)
    report = session.apply_param_edits({"s5": {"quantile": 1.5}})
    assert report.is_blocked and "G03" in report.codes()
    assert session.plan.steps[4].params["quantile"] == 1.5  # applied, then judged
    with pytest.raises(ExecutionRefusedError):
        session.execute()


# --- re-validation through the same validator ------------------------------------


def test_switching_to_full_sample_raises_g10_and_switching_back_clears_it(
    settings: Settings,
) -> None:
    session = start(settings)
    report = session.apply_param_edits({"s5": {"method": "full_sample"}})
    assert "G10" in report.codes()
    report = session.apply_param_edits({"s5": {"method": "expanding"}})
    assert "G10" not in report.codes()


def test_weekly_to_daily_warns_g14_and_runs_after_re_approval(settings: Settings) -> None:
    session = start(settings)
    report = session.apply_param_edits({"s2": {"freq": "D"}})
    assert [i.issue_id for i in report.warnings] == ["G14:s4:window", "G14:s5:min_periods"]
    session.approve([RETURN_METHOD_APPROVAL])
    assert session.execute().status == "completed"
    assert session.execution.artifacts["s4"].meta.frequency == "D"
    assert session.execution.artifacts["s4"].meta.notes[0].endswith("sqrt(252)")


def test_editing_filter_period_dates(settings: Settings) -> None:
    """Date edits go through the same validator; no date at all, or start after end, is a G03."""
    client = FakeAnthropicClient(tool_response(copy.deepcopy(PERIOD_PLAN)))
    session = Session.start("NASDAQ peak since 2021", ["NASDAQCOM"], "live", settings,
                            client=client)  # fmt: skip
    session.create_plan()
    report = session.apply_param_edits({"s2": {"start": None}})  # neither start nor end
    assert [i.issue_id for i in report.blocks] == ["G03:s2:period"]
    report = session.apply_param_edits({"s2": {"start": "2020-01-01", "end": "2019-12-31"}})
    assert [i.issue_id for i in report.blocks] == ["G03:s2:period"]
    report = session.apply_param_edits({"s2": {"end": "2020-13-01"}})  # no 13th month
    assert [i.issue_id for i in report.blocks] == ["G03:s2:end"]
    session.apply_param_edits({"s2": {"end": "2022-12-31"}})
    assert session.execute().status == "completed"
    record = session.build_record()
    assert {(c.param, c.original, c.final) for c in record.param_diff} == {
        ("start", "2021-01-01", "2020-01-01"),
        ("end", None, "2022-12-31"),
    }


# --- approvals belong to one plan version ----------------------------------------


def test_any_edit_resets_approvals_and_approvals_record_the_plan_hash(settings: Settings) -> None:
    session = start(settings)
    first_version = session.plan_hash
    session.approve([RETURN_METHOD_APPROVAL])
    assert session.approvals[0].plan_hash == first_version
    session.apply_param_edits({"s4": {"window": 26}})
    assert session.plan_hash != first_version
    assert session.approvals == []
    assert [i.issue_id for i in session.unresolved_approvals()] == [RETURN_METHOD_APPROVAL]


def test_an_approval_for_another_plan_version_does_not_count(settings: Settings) -> None:
    session = start(settings)
    stale = Approval(issue_id=RETURN_METHOD_APPROVAL, code="G12", step_id="s3",
                     message="m", approved_at="t", plan_hash="another-version")  # fmt: skip
    session.approvals.append(stale)
    assert [i.issue_id for i in session.unresolved_approvals()] == [RETURN_METHOD_APPROVAL]
    with pytest.raises(ExecutionRefusedError):
        session.execute()


def test_reset_to_the_planners_proposal(settings: Settings) -> None:
    session = start(settings)
    proposal = session.plan.model_copy(deep=True)
    session.apply_param_edits({"s2": {"freq": "D"}, "s4": {"window": 60}})
    session.reset_to_proposal()
    assert session.plan.steps == proposal.steps
    assert session.build_record().param_diff == []
    assert "G14" not in session.report.codes()


# --- lineage of edits ------------------------------------------------------------


def test_edited_run_records_the_diff_traces_it_and_replays_identically(
    settings: Settings,
) -> None:
    session = start(settings)
    session.apply_param_edits({"s4": {"window": 26}})
    session.approve([RETURN_METHOD_APPROVAL])
    session.execute()
    session.narrate()
    session.save_record()

    record = load_run(session.run_id, settings.runs_dir)
    assert record.plan_original["steps"][3]["params"]["window"] == 12  # as proposed
    assert record.plan_final["steps"][3]["params"]["window"] == 26  # as run
    (change,) = record.param_diff
    assert (change.step_id, change.param, change.original, change.final) == ("s4", "window", 12, 26)
    assert change.edited_at
    assert record.approvals[0].plan_hash == record.plan_hash

    chain = {step.step_id: step for step in session.trace("s5").chain}
    assert chain["s4"].edited == {"window": 12}  # user-edited, planner proposed 12
    assert chain["s3"].edited == {}  # planner-proposed values only

    report = replay(session.run_id, settings)  # replays plan_final, without the LLM
    assert report.all_match, report.message


# --- find_extremes: the result has one row per extreme, whatever `which` becomes ---

EXTREME_ROWS = {"max": ["max"], "min": ["min"], "both": ["max", "min"]}


@pytest.mark.parametrize(
    ("proposed", "edited"),
    [("max", "both"), ("both", "max"), ("max", "min"), ("min", "max"), ("min", "both"),
     ("both", "min"), ("both", None)],  # None: the planner proposed it, no edit
)  # fmt: skip
def test_find_extremes_which_edit_end_to_end(
    proposed: str, edited: str | None, settings: Settings
) -> None:
    """Edit find_extremes.which, then validate, execute, narrate, check, save and replay."""
    raw = canned_plan("nasdaq_volatility_peak_since_2020")  # needs the G12 approval too
    raw["steps"][5]["params"]["which"] = proposed
    request = "Show me the maximum and minimum of NASDAQ weekly volatility since 2020."
    session = Session.start(request, ["NASDAQCOM"], "live", settings,
                            client=FakeAnthropicClient(tool_response(raw)))  # fmt: skip
    session.create_plan()
    session.approve([RETURN_METHOD_APPROVAL])
    if edited is not None:
        first_version = session.plan_hash
        report = session.apply_param_edits({"s6": {"which": edited}})
        assert report.codes() == ["G12"] and not report.is_blocked  # same validator
        assert session.approvals == [] and session.plan_hash != first_version
        session.approve([RETURN_METHOD_APPROVAL])
    final = edited or proposed

    result = session.execute()
    assert result.status == "completed", result.error
    assert list(result.artifacts["s6"].value) == EXTREME_ROWS[final]
    facts = {f.fact_id: f.display for f in build_facts_sheet(session.plan, result.artifacts)}
    assert sorted(k for k in facts if k.startswith("s6.")) == sorted(
        f"s6.{row}_{column}" for row in EXTREME_ROWS[final]
        for column in ("date", "value", "tie_count")
    )  # fmt: skip

    session.mode = "mock"  # the template narrative names every extreme, all grounded
    narrative = session.narrate()
    assert narrative.grounding.grounded, narrative.grounding.ungrounded
    for row in EXTREME_ROWS[final]:
        assert f"{facts[f's6.{row}_value']} on {facts[f's6.{row}_date']}" in narrative.text
    llm_text = " ".join(
        f"The {row} volatility was {facts[f's6.{row}_value']} on {facts[f's6.{row}_date']}."
        for row in EXTREME_ROWS[final]
    )
    llm = narrate(request, session.plan, result.artifacts, [], "live", settings,
                  client=FakeAnthropicClient(text_response(llm_text)))  # fmt: skip
    assert llm.source == "llm" and llm.grounding.grounded, llm.grounding.ungrounded

    session.save_record()
    record = load_run(session.run_id, settings.runs_dir)
    assert record.plan_original["steps"][5]["params"]["which"] == proposed
    assert record.plan_final["steps"][5]["params"]["which"] == final
    changes = [(c.step_id, c.param, c.original, c.final) for c in record.param_diff]
    assert changes == ([("s6", "which", proposed, edited)] if edited else [])
    assert record.approvals[0].plan_hash == record.plan_hash
    report = replay(session.run_id, settings)
    assert report.all_match, report.message
