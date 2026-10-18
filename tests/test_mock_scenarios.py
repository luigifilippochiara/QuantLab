"""The six example workflows end to end (mock mode), plus workflow-specific checks.

One task, a market-risk analyst flagging unusual market behaviour: the flagship
analysis (1), the protections QuantLab applies (2-5) and the NASDAQ-only analysis
(6). The audit trail of example 1 (the trace) is tested in test_cli.py and
test_app.py, the lineage of example 6 in test_lineage.py.
"""

import os
import subprocess

import pytest

from quantlab.config import Settings
from quantlab.data import Catalog
from quantlab.engine import ExecutionRefusedError, Session, run_request
from quantlab.guardrails import Action, fill_defaults, validate_plan
from quantlab.lineage import replay
from quantlab.mock_scenarios import (
    DIP_REQUEST,
    FLAGSHIP_REQUEST,
    SCENARIOS,
    SCENARIOS_BY_ID,
    Scenario,
    match_scenario,
)
from quantlab.plan import Plan
from quantlab.registry import REGISTRY

from conftest import ARIMA_PLAN, SCENARIO_1_APPROVALS, SCENARIO_6_APPROVALS
from fakes import FakeAnthropicClient, tool_response


def start(scenario_id: str, settings: Settings) -> Session:
    """A mock-mode session for a scenario, with its plan created and validated."""
    scenario = SCENARIOS_BY_ID[scenario_id]
    session = Session.start(scenario.request, list(scenario.dataset_ids), "mock", settings)
    session.create_plan()
    return session


def test_examples_are_numbered_in_list_order() -> None:
    # The order of the sidebar buttons and of `cli demo`: the flagship analysis, the
    # three blocks, the look-ahead approval, then the NASDAQ-only analysis.
    assert [(s.number, s.id) for s in SCENARIOS] == [
        (1, "chf_safe_haven_success"),
        (2, "spread_frequency_mismatch_block"),
        (3, "arima_forecast_block"),
        (4, "prompt_injection_unknown_capability"),
        (5, "vix_full_sample_lookahead"),
        (6, "buy_the_dip_tech"),
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
        # Runtime guardrails may only warn here. Which ones depends on the data (E04 on
        # the committed snapshots, where the VIX starts in 1990; E05 for example 5 on the
        # synthetic VIX); the golden records pin them on both data sets.
        assert all(i.action == Action.WARN for i in session.execution.issues)
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
    ("s1", "load_series", {}, {"dataset_id": "VIXCLS"}),
    ("s2", "load_series", {}, {"dataset_id": "DEXSZUS"}),
    ("s3", "resample", {"series": "s1"}, {"freq": "W", "how": "last"}),
    ("s4", "resample", {"series": "s2"}, {"freq": "W", "how": "last"}),
    ("s5", "flag_regime", {"series": "s3"},
     {"method": "expanding", "quantile": 0.90, "min_periods": 52, "direction": "above"}),
    ("s6", "forward_return", {"series": "s4"}, {"method": "simple"}),
    ("s7", "compare_groups", {"values": "s6", "flags": "s5"}, {}),
    ("s8", "hac_mean_difference", {"outcome": "s6", "flags": "s5"}, {}),
]  # fmt: skip


def test_flagship_plan_is_exactly_the_specified_plan() -> None:
    plan = SCENARIOS_BY_ID["chf_safe_haven_success"].plan
    steps = [(s["id"], s["capability"], s["inputs"], s["params"]) for s in plan["steps"]]
    assert steps == FLAGSHIP_PLAN
    assert plan["dataset_ids"] == ["VIXCLS", "DEXSZUS"] and not plan["cannot_fulfill"]
    # The assumptions the plan declares to the analyst, including which way USD/CHF reads.
    notes = plan["planner_notes"]
    assert "the weekly VIX is its Friday close" in notes and "past levels only" in notes
    assert "simple returns for the USD/CHF outcome" in notes
    assert "a negative return means the franc strengthened" in notes


def test_example_5_is_the_flagship_with_a_full_history_threshold() -> None:
    flagship = SCENARIOS_BY_ID["chf_safe_haven_success"].plan["steps"]
    look_ahead = SCENARIOS_BY_ID["vix_full_sample_lookahead"].plan["steps"]
    differences = [
        (a["id"], name, a["params"][name], b["params"][name])
        for a, b in zip(flagship, look_ahead, strict=True)
        for name in a["params"]
        if a["params"][name] != b["params"][name]
    ]
    assert differences == [("s5", "method", "expanding", "full_sample")]


def test_requests_rely_on_these_registry_defaults() -> None:
    # The requests leave them to the registry (examples 1 and 5 their direction, every
    # example its minimum history), so a planner that omits them gets the canned plan.
    defaults = {
        ("flag_regime", "method"): "expanding",
        ("flag_regime", "quantile"): 0.90,
        ("flag_regime", "min_periods"): 52,
        ("flag_regime", "direction"): "above",
    }
    for (capability, name), expected in defaults.items():
        spec_param = REGISTRY[capability].param(name)
        assert not spec_param.required and spec_param.default == expected, (capability, name)
    # Nothing to choose for the outcome horizon or the test's lags: always one week
    # ahead, and the Newey-West rule.
    assert [p.name for p in REGISTRY["forward_return"].params] == ["method"]
    assert REGISTRY["hac_mean_difference"].params == ()


def test_flagship_validates_with_exactly_one_v10_approval(catalog: Catalog) -> None:
    raw = SCENARIOS_BY_ID["chf_safe_haven_success"].plan
    plan = fill_defaults(Plan(plan_id="p", request=FLAGSHIP_REQUEST, **raw))
    report = validate_plan(plan, catalog)
    # The outcome's return method, and nothing else: the VIX level is flagged directly
    # (no returns), both series are weekly, and the threshold uses past data only.
    assert [(i.issue_id, i.action) for i in report.issues] == [("V10:s6:return_method", "APPROVE")]
    assert "method='simple'" in report.issues[0].message


def test_flagship_runs_end_to_end_and_replays_identically(settings: Settings) -> None:
    session = run_request(FLAGSHIP_REQUEST, ["VIXCLS", "DEXSZUS"], "mock", approve=True,
                          settings=settings)  # fmt: skip
    assert session.status == "completed"
    assert [a.issue_id for a in session.approvals] == SCENARIO_1_APPROVALS
    assert session.narrative.grounding.grounded
    report = replay(session.run_id, settings)
    assert report.all_match, report.message
    original = [r.output_hash for r in session.execution.step_records]
    assert [s.replay_hash for s in report.steps] == original and len(original) == 8


def test_flagship_groups_usd_chf_returns_by_vix_flags(settings: Settings) -> None:
    session = start("chf_safe_haven_success", settings)
    session.approve(SCENARIO_1_APPROVALS)
    artifacts = session.execute().artifacts
    # The flags come from the weekly VIX level, the outcome from USD/CHF.
    assert artifacts["s3"].meta.kind == "index_level" and artifacts["s3"].meta.frequency == "W"
    assert artifacts["s5"].meta.source_dataset_ids == ["VIXCLS"]
    assert artifacts["s6"].meta.source_dataset_ids == ["DEXSZUS"]
    test = artifacts["s8"].value
    # Every flagged week but the last, which has no next-week return yet.
    flags = artifacts["s5"].value
    assert test["horizon"] == 1 and test["n_flagged"] == int(flags.iloc[:-1].sum())
    assert artifacts["s8"].meta.source_dataset_ids == ["DEXSZUS", "VIXCLS"]
    # The narrative states which way USD/CHF reads, and quotes only computed numbers.
    narrative = session.narrate()
    assert (
        "DEXSZUS is quoted as USD/CHF, in Swiss francs per US dollar: a negative return "
        "means the franc strengthened against the dollar" in narrative.text
    )
    assert narrative.grounding.grounded and narrative.issues == []


def test_flagship_demo_edit_dataset_to_a_yield_is_blocked(settings: Settings) -> None:
    session = start("chf_safe_haven_success", settings)
    report = session.apply_param_edits({"s2": {"dataset_id": "DGS10"}})
    # A legal edit (a parameter value), which the re-validation blocks: a forward
    # "return" needs a price level, and a yield is a rate. (V07 RETURNS_ON_RATES needs
    # a compute_returns step: the same edit on example 6 shows it.)
    assert session.plan.steps[1].params["dataset_id"] == "DGS10"
    assert [(i.issue_id, i.name) for i in report.blocks] == [("V04:s6:series", "TYPE_MISMATCH")]
    with pytest.raises(ExecutionRefusedError):
        session.execute()


def test_flagship_demo_edit_quantile_out_of_range_is_blocked(settings: Settings) -> None:
    session = start("chf_safe_haven_success", settings)
    report = session.apply_param_edits({"s5": {"quantile": 1.5}})
    (v03,) = report.blocks
    assert (v03.code, v03.name, v03.step_id) == ("V03", "INVALID_PARAM", "s5")
    assert "'quantile'=1.5 is outside [0.01, 0.99]" in v03.message
    with pytest.raises(ExecutionRefusedError):
        session.execute()


# --- example 6, buy the dip: the bottom tail of past weekly returns ---------------

# (step id, capability, inputs, params): the plan example 6's request specifies.
DIP_PLAN = [
    ("s1", "load_series", {}, {"dataset_id": "NASDAQCOM"}),
    ("s2", "resample", {"series": "s1"}, {"freq": "W", "how": "last"}),
    ("s3", "compute_returns", {"series": "s2"}, {"method": "simple", "periods": 1}),
    ("s4", "flag_regime", {"series": "s3"},
     {"method": "expanding", "quantile": 0.10, "min_periods": 52, "direction": "below"}),
    ("s5", "forward_return", {"series": "s2"}, {"method": "simple"}),
    ("s6", "compare_groups", {"values": "s5", "flags": "s4"}, {}),
    ("s7", "hac_mean_difference", {"outcome": "s5", "flags": "s4"}, {}),
]  # fmt: skip


def test_scenario_6_plan_is_exactly_the_specified_plan() -> None:
    plan = SCENARIOS_BY_ID["buy_the_dip_tech"].plan
    steps = [(s["id"], s["capability"], s["inputs"], s["params"]) for s in plan["steps"]]
    assert steps == DIP_PLAN
    assert plan["dataset_ids"] == ["NASDAQCOM"] and not plan["cannot_fulfill"]
    notes = plan["planner_notes"]
    assert "below the 10th percentile of past weekly returns only" in notes
    assert "simple weekly returns" in notes


def test_scenario_6_validates_with_exactly_two_v10_approvals(catalog: Catalog) -> None:
    raw = SCENARIOS_BY_ID["buy_the_dip_tech"].plan
    plan = fill_defaults(Plan(plan_id="p", request=DIP_REQUEST, **raw))
    report = validate_plan(plan, catalog)
    assert [(i.issue_id, i.action) for i in report.issues] == [
        ("V10:s3:return_method", "APPROVE"),  # simple returns, the ones flagged
        ("V10:s5:return_method", "APPROVE"),  # simple returns for the outcome
    ]  # nothing else: no WARN, no BLOCK, no look-ahead (an expanding threshold)


def test_scenario_6_runs_end_to_end_and_replays_identically(settings: Settings) -> None:
    session = run_request(DIP_REQUEST, ["NASDAQCOM"], "mock", approve=True, settings=settings)
    assert session.status == "completed" and not session.execution.issues
    assert [a.issue_id for a in session.approvals] == SCENARIO_6_APPROVALS
    assert session.narrative.grounding.grounded
    report = replay(session.run_id, settings)
    assert report.all_match, report.message
    original = [r.output_hash for r in session.execution.step_records]
    assert [s.replay_hash for s in report.steps] == original and len(original) == 7


def test_scenario_6_flags_the_biggest_drops_from_past_data_only(settings: Settings) -> None:
    session = start("buy_the_dip_tech", settings)
    session.approve(SCENARIO_6_APPROVALS)
    artifacts = session.execute().artifacts
    returns, flags = artifacts["s3"].value, artifacts["s4"].value
    # Each dip is below the 10% quantile of the returns BEFORE it (no look-ahead).
    threshold = returns.shift(1).expanding(min_periods=52).quantile(0.10)
    assert flags.any() and (returns[flags.index][flags] < threshold[flags.index][flags]).all()
    assert (returns[flags.index][flags] < 0).all()  # every dip is a drop
    # The narrative answers the question directly, in grounded numbers.
    narrative = session.narrate()
    test = artifacts["s7"].value
    direction = "higher" if test["diff_means"] > 0 else "lower"
    assert f"Next-week returns were {direction} after flagged weeks than after" in narrative.text
    assert narrative.grounding.grounded and narrative.issues == []


def test_scenario_6_demo_edit_dataset_to_a_yield_blocks_returns_on_rates(
    settings: Settings,
) -> None:
    session = start("buy_the_dip_tech", settings)
    report = session.apply_param_edits({"s1": {"dataset_id": "DGS10"}})
    # A legal edit, which the re-validation blocks: "returns" of a yield are
    # meaningless (V07), and a forward return needs a price level (V04).
    assert session.plan.steps[0].params["dataset_id"] == "DGS10"
    assert [i.issue_id for i in report.blocks] == ["V07:s3", "V04:s5:series"]
    with pytest.raises(ExecutionRefusedError):
        session.execute()


def test_scenario_5_full_sample_threshold_needs_approval_then_runs(settings: Settings) -> None:
    session = start("vix_full_sample_lookahead", settings)
    # The look-ahead approval, and the flagship's return-method approval: nothing else.
    assert [(i.issue_id, i.action) for i in session.report.issues] == [
        ("V09:s5", "APPROVE"),
        ("V10:s6:return_method", "APPROVE"),
    ]
    v09 = session.report.issues[0]
    assert "information from the future" in v09.message
    assert "It also invalidates the significance test in s8" in v09.message
    assert "method='expanding'" in v09.suggestion and "descriptive labelling" in v09.suggestion
    with pytest.raises(ExecutionRefusedError, match="V09:s5"):
        session.execute()  # not without the human's decision
    session.approve(["V09:s5", "V10:s6:return_method"])
    assert session.execute().status == "completed"
    flags = session.execution.artifacts["s5"]
    # Weekly flags above the whole history's 90th percentile: about 10% of the weeks.
    assert flags.meta.frequency == "W" and 0.09 < flags.value.mean() < 0.11
    assert "uses future data" in flags.meta.notes[0]


def test_scenario_3_refusal_lists_what_the_registry_can_do(settings: Settings) -> None:
    session = start("arima_forecast_block", settings)
    reason = session.plan.refusal_reason
    assert reason.startswith("No registered capability fits a model or makes a forecast")
    assert "ARIMA forecast of the S&P 500" in reason
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


def test_scenario_2_frequency_mismatch_blocks_before_anything_runs(settings: Settings) -> None:
    session = start("spread_frequency_mismatch_block", settings)
    # The frequency mismatch, and the look-ahead approval of the full-history threshold.
    assert [(i.issue_id, i.action) for i in session.report.issues] == [
        ("V08:s4", "BLOCK"),
        ("V09:s5", "APPROVE"),
    ]
    v08 = session.report.blocks[0]
    assert "'a' is weekly, 'b' is daily" in v08.message
    assert "Resample DGS2 to weekly" in v08.suggestion
    assert session.status == "blocked" and not session.can_execute()[0]
    with pytest.raises(ExecutionRefusedError):
        session.execute()


def test_scenario_2_asks_for_the_bottom_tail_of_the_spread() -> None:
    flags = SCENARIOS_BY_ID["spread_frequency_mismatch_block"].plan["steps"][4]
    assert (flags["capability"], flags["inputs"]) == ("flag_regime", {"series": "s4"})
    assert flags["params"] == {"method": "full_sample", "quantile": 0.10, "min_periods": 52,
                               "direction": "below"}  # fmt: skip


def test_scenario_2_runs_once_both_yields_are_daily(settings: Settings) -> None:
    # The fix the app allows (steps are locked): the 10-year kept daily, like the 2-year.
    session = start("spread_frequency_mismatch_block", settings)
    report = session.apply_param_edits({"s2": {"freq": "D"}})
    assert [i.issue_id for i in report.issues] == ["V09:s5"]  # no more V08
    session.approve(["V09:s5"])
    execution = session.execute()
    assert execution.status == "completed"
    spread = execution.artifacts["s4"].value
    flags = execution.artifacts["s5"].value
    # The bottom tail of the whole history: below the 10% quantile, about 10% of the days.
    assert flags.any() and (spread[flags] < spread.quantile(0.10)).all()
    assert 0.09 < flags.mean() < 0.11
