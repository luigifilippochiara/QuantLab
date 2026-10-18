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
from quantlab.mock_scenarios import (
    FLAGSHIP_REQUEST,
    SCENARIOS,
    SCENARIOS_BY_ID,
    Scenario,
    match_scenario,
)
from quantlab.plan import Plan
from quantlab.registry import REGISTRY

from conftest import ARIMA_PLAN, SCENARIO_1_APPROVALS
from fakes import FakeAnthropicClient, tool_response


def start(scenario_id: str, settings: Settings) -> Session:
    """A mock-mode session for a scenario, with its plan created and validated."""
    scenario = SCENARIOS_BY_ID[scenario_id]
    session = Session.start(scenario.request, list(scenario.dataset_ids), "mock", settings)
    session.create_plan()
    return session


def test_examples_are_numbered_in_list_order() -> None:
    # The order of the sidebar buttons and of `cli demo`: the core analysis, then
    # the blocks, and the look-ahead approval last.
    assert [(s.number, s.id) for s in SCENARIOS] == [
        (1, "volatility_flag_success"),
        (2, "spread_frequency_mismatch_block"),
        (3, "arima_forecast_block"),
        (4, "prompt_injection_unknown_capability"),
        (5, "vix_full_sample_lookahead"),
    ]
    assert all(s.label.startswith(f"{s.number} · ") for s in SCENARIOS)


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


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
def test_every_request_matches_only_its_own_scenario(scenario: Scenario) -> None:
    assert match_scenario(scenario.request) is scenario  # by exact text
    lowered = scenario.request.lower()
    by_keywords = [
        other.id
        for other in SCENARIOS
        if all(any(word in lowered for word in group) for group in other.keywords)
    ]
    assert by_keywords == [scenario.id]  # and by keywords, whatever the matching order


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


def test_flagship_validates_with_exactly_two_v10_approvals(catalog: Catalog) -> None:
    raw = SCENARIOS_BY_ID["volatility_flag_success"].plan
    plan = fill_defaults(Plan(plan_id="p", request=FLAGSHIP_REQUEST, **raw))
    report = validate_plan(plan, catalog)
    assert [(i.issue_id, i.action) for i in report.issues] == [
        ("V10:s3:return_method", "APPROVE"),  # log returns for the volatility
        ("V10:s6:return_method", "APPROVE"),  # simple returns for the outcome
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


def test_scenario_1_demo_edit_full_sample_raises_v09_for_the_test(settings: Settings) -> None:
    session = start("volatility_flag_success", settings)
    report = session.apply_param_edits({"s5": {"method": "full_sample"}})
    v09 = next(i for i in report.approvals_required if i.code == "V09")
    assert "invalidates the significance test in s8" in v09.message
    session.approve(SCENARIO_1_APPROVALS)
    with pytest.raises(ExecutionRefusedError, match="V09:s5"):
        session.execute()
    session.approve(["V09:s5"])
    assert session.execute().status == "completed"


def test_scenario_1_demo_edit_dataset_to_a_yield_blocks_returns_on_rates(
    settings: Settings,
) -> None:
    session = start("volatility_flag_success", settings)
    report = session.apply_param_edits({"s1": {"dataset_id": "DGS10"}})
    # A legal edit: a parameter value, applied to the plan...
    assert not any(i.message.startswith("Edit rejected") for i in report.issues)
    assert session.plan.steps[0].params["dataset_id"] == "DGS10"
    # ...that the re-validation blocks: "returns" of a yield are meaningless.
    v07 = next(i for i in report.blocks if i.code == "V07")
    assert (v07.name, v07.step_id) == ("RETURNS_ON_RATES", "s3")
    assert "difference" in v07.suggestion
    # forward_return (s6) needs a price level too.
    assert [i.issue_id for i in report.blocks] == ["V07:s3", "V04:s6:series"]
    with pytest.raises(ExecutionRefusedError):
        session.execute()


def test_scenario_2_full_sample_threshold_needs_approval_then_runs(settings: Settings) -> None:
    session = start("vix_full_sample_lookahead", settings)
    (v09,) = session.report.issues  # the look-ahead approval, and nothing else
    assert (v09.issue_id, v09.action) == ("V09:s2", "APPROVE")
    assert "information from the future" in v09.message
    assert "method='expanding'" in v09.suggestion and "descriptive labelling" in v09.suggestion
    with pytest.raises(ExecutionRefusedError, match="V09:s2"):
        session.execute()  # not without the human's decision
    session.approve(["V09:s2"])
    assert session.execute().status == "completed"
    flags = session.execution.artifacts["s2"]
    # Daily flags above the whole history's 95th percentile: about 5% of the days.
    assert flags.meta.frequency == "D" and 0.04 < flags.value.mean() < 0.06
    assert "uses future data" in flags.meta.notes[0]


def test_scenario_3_refusal_lists_what_the_registry_can_do(settings: Settings) -> None:
    session = start("arima_forecast_block", settings)
    reason = session.plan.refusal_reason
    assert reason.startswith("No registered capability fits a model or makes a forecast")
    assert "ARIMA forecast of the USD/CHF exchange rate" in reason
    assert all(name in reason for name in REGISTRY)  # the list comes from the registry
    assert session.status == "blocked" and session.execution is None


def test_scenario_3_wall_holds_even_if_the_planner_misbehaves(settings: Settings) -> None:
    """A planner that emits an 'arima' step instead of refusing is still blocked."""
    request = SCENARIOS_BY_ID["arima_forecast_block"].request
    client = FakeAnthropicClient(tool_response(ARIMA_PLAN))
    session = run_request(request, ["NASDAQCOM"], "live", settings=settings, client=client)
    assert session.report.codes() == ["V02"]
    assert session.status == "blocked" and session.execution is None


def test_scenario_4_injected_plan_is_blocked_and_never_runs(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The simulated planner obeys the injection; the validator blocks it (V02); nothing runs."""
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

    scenario = SCENARIOS_BY_ID["prompt_injection_unknown_capability"]
    session = run_request(scenario.request, list(scenario.dataset_ids), "mock", settings=settings)

    # A well-formed plan (no V01) that follows the injection, blocked by V02 alone.
    assert [s.capability for s in session.plan.steps] == ["load_series", "run_python"]
    assert session.plan.steps[1].params == {"code": "import os; os.system('ls')"}
    assert [(i.issue_id, i.action) for i in session.report.issues] == [("V02:s2", "BLOCK")]
    # Blocked before approval: the executor is never reached, no step ever runs.
    assert session.status == "blocked" and session.execution is None and not session.approvals
    assert "run_python" not in REGISTRY
    assert shell_commands == []  # os.system / os.popen never called
    # The only processes are the run record's fixed, read-only git queries (commit
    # hash and dirty flag); nothing from the request ever reaches a process.
    assert processes and all(args[0] == "git" for args in processes)
    assert not any("ls" in args or "os.system" in " ".join(args) for args in processes)


def test_scenario_5_frequency_mismatch_blocks_before_anything_runs(settings: Settings) -> None:
    session = start("spread_frequency_mismatch_block", settings)
    (v08,) = session.report.issues  # the frequency mismatch, and nothing else
    assert (v08.issue_id, v08.action) == ("V08:s4", "BLOCK")
    assert "'a' is weekly, 'b' is daily" in v08.message
    assert "Resample DGS2 to weekly" in v08.suggestion
    assert session.status == "blocked" and not session.can_execute()[0]
    with pytest.raises(ExecutionRefusedError):
        session.execute()
