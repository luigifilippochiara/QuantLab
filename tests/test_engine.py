"""Engine: the full plan -> validate -> edit -> approve -> execute -> narrate -> record flow."""

import pytest

from quantlab.config import Settings
from quantlab.engine import ExecutionRefusedError, Session, run_request
from quantlab.lineage import load_run, replay
from quantlab.mock_scenarios import SCENARIOS_BY_ID
from quantlab.monitoring import kpis, read_events

from conftest import SCENARIO_1_APPROVALS
from fakes import FakeAnthropicClient

SUCCESS = SCENARIOS_BY_ID["volatility_flag_success"]  # scenario 1


def start(settings: Settings, scenario_id: str = SUCCESS.id) -> Session:
    """A session for a demo scenario in mock mode, with its plan created."""
    scenario = SCENARIOS_BY_ID[scenario_id]
    session = Session.start(scenario.request, list(scenario.dataset_ids), "mock", settings)
    session.create_plan()
    return session


def test_end_to_end_scenario_1_with_manifest_and_replay(settings: Settings) -> None:
    session = start(settings)
    assert session.status == "ready"
    assert [i.issue_id for i in session.unresolved_approvals()] == SCENARIO_1_APPROVALS

    with pytest.raises(ExecutionRefusedError, match="Approval required"):
        session.execute()

    session.approve(SCENARIO_1_APPROVALS)
    assert session.execute().status == "completed"
    narrative = session.narrate()
    assert narrative.grounding.grounded
    manifest = session.save_record()

    record = load_run(session.run_id, settings.runs_dir)
    assert manifest.exists() and record.status == "completed"
    assert [a.issue_id for a in record.approvals] == SCENARIO_1_APPROVALS
    assert record.llm.provider == "mock" and record.registry_version
    assert record.plan_original["steps"][4]["params"]["quantile"] == 0.9  # as proposed
    assert record.plan_final["steps"] == record.plan_original["steps"]  # no hidden defaults
    assert set(record.stage_timings_ms) == {"planning", "validation", "execution", "narration"}

    report = replay(session.run_id, settings)
    assert report.all_match, report.message


def test_param_edit_revalidates_and_clears_approvals(settings: Settings) -> None:
    session = start(settings)
    session.approve(SCENARIO_1_APPROVALS)
    report = session.apply_param_edits({"s3": {"method": "simple"}, "s4": {"window": 26}})
    assert not report.is_blocked
    assert session.approvals == []  # the approved plan changed: approve again
    assert "method='simple'" in session.unresolved_approvals()[0].message
    assert [(e.step_id, e.param, e.old, e.new) for e in session.edits] == [
        ("s3", "method", "log", "simple"),
        ("s4", "window", 12, 26),
    ]
    session.approve(SCENARIO_1_APPROVALS)
    session.execute()
    assert session.execution.artifacts["s3"].meta.return_method == "simple"


def test_invalid_edit_blocks_execution(settings: Settings) -> None:
    session = start(settings)
    report = session.apply_param_edits({"s4": {"window": 1}})
    assert report.is_blocked and "G03" in report.codes()
    allowed, reason = session.can_execute()
    assert not allowed and "G03" in reason
    with pytest.raises(ExecutionRefusedError):
        session.execute()
    # Fixing the edit unblocks it again.
    assert not session.apply_param_edits({"s4": {"window": 12}}).is_blocked


def test_only_open_approval_issues_can_be_approved(settings: Settings) -> None:
    session = start(settings)
    with pytest.raises(ValueError):
        session.approve(["G09:s5"])  # not an issue of scenario 1's plan
    # An edit to a step that does not exist is rejected (see test_editing.py).
    assert session.apply_param_edits({"s99": {"window": 3}}).is_blocked


def test_blocked_scenario_is_recorded(settings: Settings) -> None:
    session = start(settings, "arima_forecast_block")
    assert session.status == "blocked"
    record = load_run(session.run_id, settings.runs_dir)
    assert record.status == "blocked" and "G11" in record.error
    events = read_events(settings.events_path)
    assert events[-1]["event_type"] == "run_failed"
    assert events[-1]["details"]["status"] == "blocked"


def test_planner_failure_is_a_recorded_g01(settings: Settings) -> None:
    client = FakeAnthropicClient(error=RuntimeError("service unavailable"))
    session = Session.start("anything", ["NASDAQCOM"], "live", settings, client)
    report = session.create_plan()
    assert report.codes() == ["G01"] and session.status == "blocked"
    record = load_run(session.run_id, settings.runs_dir)
    assert "service unavailable" in record.error
    assert record.llm.provider == "anthropic"


def test_over_budget_request_never_reaches_the_planner(settings: Settings) -> None:
    client = FakeAnthropicClient(error=AssertionError("must not be called"))
    session = Session.start("x" * 5000, ["NASDAQCOM"], "live", settings, client)
    assert session.create_plan().codes() == ["G05"]
    assert client.calls == []


def test_run_request_without_approval_executes_nothing(settings: Settings) -> None:
    session = run_request(SUCCESS.request, list(SUCCESS.dataset_ids), "mock", settings=settings)
    assert session.status == "blocked" and session.execution is None
    assert "Approval required" in load_run(session.run_id, settings.runs_dir).error


def test_monitoring_sees_every_stage(settings: Settings) -> None:
    run_request(SUCCESS.request, list(SUCCESS.dataset_ids), "mock", True, settings)
    start(settings, "prompt_injection_block")
    events = read_events(settings.events_path)
    types = {e["event_type"] for e in events}
    assert {"request_received", "plan_generated", "plan_validated", "approval_granted",
            "execution_started", "step_completed", "execution_finished",
            "narrative_checked", "run_completed", "run_failed"} <= types  # fmt: skip
    summary = kpis(events)
    assert (summary["n_runs"], summary["n_completed"], summary["n_blocked"]) == (2, 1, 1)
