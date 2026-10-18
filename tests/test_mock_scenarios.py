"""The five example workflows end to end (mock mode), plus workflow-specific checks.

One task, a market-risk analyst flagging unusual market behaviour: the core
analysis (1) and the protections QuantLab applies (2-5). The audit trail of
example 1 (the trace) is tested in test_lineage.py and test_cli.py.
"""

import os
import subprocess

import pytest

from quantlab.config import Settings
from quantlab.data import Catalog
from quantlab.engine import ExecutionRefusedError, Session, run_request
from quantlab.guardrails import fill_defaults, validate_plan
from quantlab.lineage import replay
from quantlab.mock_scenarios import FLAGSHIP_REQUEST, SCENARIOS, SCENARIOS_BY_ID, Scenario
from quantlab.plan import Plan
from quantlab.registry import REGISTRY

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


# --- the flagship (example 1): fully specified, so mock and live should agree -----

# (step id, capability, inputs, params): the plan the flagship request specifies.
FLAGSHIP_PLAN = [
    ("s1", "load_series", {}, {"dataset_id": "NASDAQCOM"}),
    ("s2", "resample", {"series": "s1"}, {"freq": "W", "how": "last"}),
    ("s3", "compute_returns", {"series": "s2"}, {"method": "log", "periods": 1}),
    ("s4", "rolling_volatility", {"series": "s3"}, {"window": 12, "annualize": True}),
    ("s5", "flag_regime", {"series": "s4"},
     {"method": "expanding", "quantile": 0.90, "min_periods": 52}),
    ("s6", "forward_return", {"series": "s2"}, {"method": "simple"}),
    ("s7", "compare_groups", {"values": "s6", "flags": "s5"}, {}),
    ("s8", "hac_mean_difference", {"outcome": "s6", "flags": "s5"}, {}),
]  # fmt: skip


def test_flagship_plan_is_exactly_the_specified_plan() -> None:
    plan = SCENARIOS_BY_ID["volatility_flag_success"].plan
    steps = [(s["id"], s["capability"], s["inputs"], s["params"]) for s in plan["steps"]]
    assert steps == FLAGSHIP_PLAN
    assert plan["dataset_ids"] == ["NASDAQCOM"] and not plan["cannot_fulfill"]
    # The assumptions the plan declares to the analyst.
    notes = plan["planner_notes"]
    assert "log returns for the volatility" in notes and "square-root-of-time" in notes
    assert "simple returns for the outcome" in notes and "half the variance" in notes
    assert "past volatility only" in notes


def test_flagship_request_relies_on_these_registry_defaults() -> None:
    # The request does not state them, so a planner that omits them gets the canned plan.
    defaults = {
        ("rolling_volatility", "annualize"): True,
        ("flag_regime", "method"): "expanding",
        ("flag_regime", "quantile"): 0.90,
        ("flag_regime", "min_periods"): 52,
    }
    for (capability, name), expected in defaults.items():
        spec_param = REGISTRY[capability].param(name)
        assert not spec_param.required and spec_param.default == expected, (capability, name)
    # Nothing to choose for the outcome horizon or the test's lags: always one week
    # ahead, and the Newey-West rule.
    assert [p.name for p in REGISTRY["forward_return"].params] == ["method"]
    assert REGISTRY["hac_mean_difference"].params == ()


def test_flagship_validates_with_exactly_two_g10_approvals(catalog: Catalog) -> None:
    raw = SCENARIOS_BY_ID["volatility_flag_success"].plan
    plan = fill_defaults(Plan(plan_id="p", request=FLAGSHIP_REQUEST, **raw))
    report = validate_plan(plan, catalog)
    assert [(i.issue_id, i.severity) for i in report.issues] == [
        ("G10:s3:return_method", "APPROVAL"),  # log returns for the volatility
        ("G10:s6:return_method", "APPROVAL"),  # simple returns for the outcome
    ]  # nothing else: no WARN, no BLOCK
    assert "method='log'" in report.issues[0].message
    assert "method='simple'" in report.issues[1].message


def test_flagship_runs_end_to_end_and_replays_identically(settings: Settings) -> None:
    session = run_request(FLAGSHIP_REQUEST, ["NASDAQCOM"], "mock", approve=True,
                          settings=settings)  # fmt: skip
    assert session.status == "completed" and not session.execution.issues
    assert [a.issue_id for a in session.approvals] == SCENARIO_1_APPROVALS
    assert session.narrative.grounding.grounded
    report = replay(session.run_id, settings)
    assert report.all_match, report.message
    original = [r.output_hash for r in session.execution.step_records]
    assert [s.replay_hash for s in report.steps] == original and len(original) == 8


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
    assert test["maxlags_used"] == 7  # floor(4 * (T/100)^(2/9)) for this sample size
    narrative = session.narrate()
    assert "statistically significant difference" in narrative.text
    assert f"form {test['n_episodes_flagged']} distinct episodes" in narrative.text
    assert narrative.grounding.grounded


def test_scenario_1_demo_edit_full_sample_raises_g09_for_the_test(settings: Settings) -> None:
    session = start("volatility_flag_success", settings)
    report = session.apply_param_edits({"s5": {"method": "full_sample"}})
    g09 = next(i for i in report.approvals_required if i.code == "G09")
    assert "invalidates the significance test in s8" in g09.message
    session.approve(SCENARIO_1_APPROVALS)
    with pytest.raises(ExecutionRefusedError, match="G09:s5"):
        session.execute()
    session.approve(["G09:s5"])
    assert session.execute().status == "completed"


def test_scenario_2_needs_both_approvals(settings: Settings) -> None:
    session = start("volatility_flag_lookahead", settings)
    g09 = next(i for i in session.report.issues if i.code == "G09")
    assert g09.severity == "APPROVAL" and "information from the future" in g09.message
    assert "method='expanding'" in g09.suggestion
    session.approve([RETURN_METHOD_APPROVAL])
    with pytest.raises(ExecutionRefusedError, match="G09:s5"):
        session.execute()  # one approval is not enough
    session.approve(["G09:s5"])
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
    g08 = next(i for i in session.report.issues if i.code == "G08")
    assert g08.severity == "BLOCK" and g08.step_id == "s6"
    assert "Resample VIXCLS to weekly" in g08.suggestion
    assert session.status == "blocked" and not session.can_execute()[0]
    with pytest.raises(ExecutionRefusedError):
        session.execute()
