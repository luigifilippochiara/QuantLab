"""The five example workflows end to end (mock mode), plus workflow-specific checks.

One task, a market-risk analyst flagging unusual market behaviour: the core
analysis (1) and the protections QuantLab applies (2-5). The audit trail of
example 1 (the trace) is tested in test_lineage.py and test_cli.py.
"""

import os
import subprocess

import pytest

from quantlab.config import Settings
from quantlab.engine import ExecutionRefusedError, Session, run_request
from quantlab.mock_scenarios import (
    CANNED_REQUESTS,
    SCENARIOS,
    SCENARIOS_BY_ID,
    CannedRequest,
    Scenario,
)

from conftest import ARIMA_PLAN, RETURN_METHOD_APPROVAL, SCENARIO_1_APPROVALS
from fakes import FakeAnthropicClient, tool_response


def start(scenario_id: str, settings: Settings) -> Session:
    """A mock-mode session for a scenario, with its plan created and validated."""
    scenario = SCENARIOS_BY_ID[scenario_id]
    session = Session.start(scenario.request, list(scenario.dataset_ids), "mock", settings)
    session.create_plan()
    return session


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
def test_scenario_end_to_end(scenario: Scenario, settings: Settings) -> None:
    session = run_request(
        scenario.request,
        list(scenario.dataset_ids),
        "mock",
        approve=scenario.auto_approve,
        settings=settings,
    )
    assert session.status == scenario.expected_status, session.build_record().error
    assert session.report.codes() == sorted(scenario.expected_codes)
    if scenario.expected_status == "completed":
        assert not session.execution.issues  # no runtime guardrail fired
        assert session.narrative.grounding.grounded
    else:
        assert session.execution is None  # nothing ran


@pytest.mark.parametrize("canned", CANNED_REQUESTS, ids=lambda c: c.id)
def test_canned_request_end_to_end(canned: CannedRequest, settings: Settings) -> None:
    """The mock-mode plans for filter_period and find_extremes run and narrate cleanly."""
    session = run_request(canned.request, list(canned.dataset_ids), "mock", approve=True,
                          settings=settings)  # fmt: skip
    assert session.status == "completed", session.build_record().error
    assert session.report.codes() == sorted(canned.expected_codes)
    assert not session.execution.issues
    assert session.narrative.grounding.grounded, session.narrative.grounding.ungrounded


def test_scenario_1_needs_the_return_method_approvals_then_tests(settings: Settings) -> None:
    session = start("volatility_flag_success", settings)
    assert not session.report.is_blocked and session.report.warnings == []
    assert [i.issue_id for i in session.report.approvals_required] == SCENARIO_1_APPROVALS
    session.approve(SCENARIO_1_APPROVALS)
    session.execute()
    flags = session.execution.artifacts["s5"]
    assert flags.meta.frequency == "W" and flags.value.any()  # flagged weeks to shade
    test = session.execution.artifacts["s8"].value
    assert 0 < test["p_value"] < 1 and test["horizon"] == 1
    assert test["maxlags_used"] >= 1  # the Newey-West rule for this sample size
    narrative = session.narrate()
    assert "statistically significant difference" in narrative.text
    assert f"form {test['n_episodes_flagged']} distinct episodes" in narrative.text
    assert narrative.grounding.grounded


def test_scenario_1_demo_edit_four_week_horizon_warns_g11(settings: Settings) -> None:
    session = start("volatility_flag_success", settings)
    report = session.apply_param_edits({"s6": {"horizon": 4}})
    assert [i.issue_id for i in report.warnings] == ["G11:s6"] and not report.is_blocked
    session.approve(SCENARIO_1_APPROVALS)
    assert session.execute().status == "completed"
    test = session.execution.artifacts["s8"].value
    assert test["horizon"] == 4 and test["maxlags_used"] >= 3  # covers the 3-week overlap
    narrative = session.narrate()
    assert "(G11)" in narrative.text and narrative.grounding.grounded


def test_scenario_1_demo_edit_full_sample_raises_g10_for_the_test(settings: Settings) -> None:
    session = start("volatility_flag_success", settings)
    report = session.apply_param_edits({"s5": {"method": "full_sample"}})
    g10 = next(i for i in report.approvals_required if i.code == "G10")
    assert "invalidates the significance test in s8" in g10.message
    session.approve(SCENARIO_1_APPROVALS)
    with pytest.raises(ExecutionRefusedError, match="G10:s5"):
        session.execute()
    session.approve(["G10:s5"])
    assert session.execute().status == "completed"


def test_scenario_2_needs_both_approvals(settings: Settings) -> None:
    session = start("volatility_flag_lookahead", settings)
    g10 = next(i for i in session.report.issues if i.code == "G10")
    assert g10.severity == "APPROVAL" and "information from the future" in g10.message
    assert "method='expanding'" in g10.suggestion
    session.approve([RETURN_METHOD_APPROVAL])
    with pytest.raises(ExecutionRefusedError, match="G10:s5"):
        session.execute()  # one approval is not enough
    session.approve(["G10:s5"])
    assert session.execute().status == "completed"


def test_scenario_3_refusal_explains_what_is_possible(settings: Settings) -> None:
    session = start("arima_forecast_block", settings)
    reason = session.plan.refusal_reason
    assert "no registered capability for model fitting or prediction" in reason
    assert "an ARIMA forecast" in reason
    assert "rolling volatility" in reason  # what the registry can do instead
    assert session.status == "blocked" and session.execution is None


def test_scenario_3_wall_holds_even_if_the_planner_misbehaves(settings: Settings) -> None:
    """A planner that emits an 'arima' step instead of refusing is still blocked."""
    request = SCENARIOS_BY_ID["arima_forecast_block"].request
    client = FakeAnthropicClient(tool_response(ARIMA_PLAN))
    session = run_request(request, ["NASDAQCOM"], "live", settings=settings, client=client)
    assert session.report.codes() == ["G02"]
    assert session.status == "blocked" and session.execution is None


def test_scenario_4_injection_never_reaches_a_shell(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    shell_commands: list[object] = []
    processes: list[list[str]] = []
    real_popen = subprocess.Popen

    class SpyPopen(real_popen):  # records every process started, then starts it
        def __init__(self, args, *more, **kwargs):
            processes.append(list(args) if not isinstance(args, str) else [args])
            super().__init__(args, *more, **kwargs)

    monkeypatch.setattr(os, "system", lambda command: shell_commands.append(command) or 0)
    monkeypatch.setattr(os, "popen", lambda *args, **kwargs: shell_commands.append(args))
    monkeypatch.setattr(subprocess, "Popen", SpyPopen)  # subprocess.run goes through Popen

    scenario = SCENARIOS_BY_ID["prompt_injection_block"]
    session = run_request(scenario.request, list(scenario.dataset_ids), "mock", settings=settings)

    assert session.status == "blocked" and session.execution is None
    assert "treated as data" in session.plan.refusal_reason
    assert shell_commands == []  # os.system / os.popen never called
    # The only processes are the run record's fixed, read-only git queries (commit
    # hash and dirty flag); nothing from the request ever reaches a process.
    assert processes and all(args[0] == "git" for args in processes)
    assert not any("ls" in args or "os.system" in " ".join(args) for args in processes)


def test_scenario_5_frequency_mismatch_blocks_before_anything_runs(settings: Settings) -> None:
    session = start("frequency_mismatch_block", settings)
    g09 = next(i for i in session.report.issues if i.code == "G09")
    assert g09.severity == "BLOCK" and g09.step_id == "s6"
    assert "Resample VIXCLS to weekly" in g09.suggestion
    assert session.status == "blocked" and not session.can_execute()[0]
    with pytest.raises(ExecutionRefusedError):
        session.execute()
