"""The five demo scenarios end to end (mock mode), plus scenario-specific checks.

One use case: a market-risk analyst flagging unusual market behaviour. It works
(1), it is smart (2), it is safe (3, 4, 5); scenario 6 (the trace) is tested in
test_lineage.py and test_cli.py.
"""

import os
import subprocess

import pytest

from quantlab.config import Settings
from quantlab.engine import ExecutionRefusedError, Session, run_request
from quantlab.scenarios import SCENARIOS, SCENARIOS_BY_ID, Scenario

from conftest import RANDOM_FOREST_PLAN, RETURN_METHOD_APPROVAL
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


def test_scenario_1_needs_one_approval_then_flags_weeks(settings: Settings) -> None:
    session = start("volatility_flag_success", settings)
    assert not session.report.is_blocked
    assert [i.issue_id for i in session.report.approvals_required] == [RETURN_METHOD_APPROVAL]
    session.approve([RETURN_METHOD_APPROVAL])
    session.execute()
    flags = session.execution.artifacts["s5"]
    assert flags.meta.frequency == "W" and flags.value.any()  # flagged weeks to shade


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
    session = start("random_forest_block", settings)
    reason = session.plan.refusal_reason
    assert "no registered capability for model fitting or prediction" in reason
    assert "rolling volatility" in reason  # what the registry can do instead
    assert session.status == "blocked" and session.execution is None


def test_scenario_3_wall_holds_even_if_the_planner_misbehaves(settings: Settings) -> None:
    """A planner that emits a 'random_forest' step instead of refusing is still blocked."""
    request = SCENARIOS_BY_ID["random_forest_block"].request
    client = FakeAnthropicClient(tool_response(RANDOM_FOREST_PLAN))
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
