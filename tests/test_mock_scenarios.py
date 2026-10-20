"""The five example workflows end to end (mock mode), plus workflow-specific checks.

One task, a market-risk analyst flagging unusual market behaviour: two complete
analyses (1, 5), a blocked request that can be fixed and re-run (2), and an
out-of-scope and an unsafe request QuantLab stops (3, 4). The audit trail of example
1 (the trace) is tested in test_cli.py and test_app.py.
"""

import dataclasses
import os
import subprocess
from pathlib import Path

import pytest

from quantlab.config import PROJECT_ROOT, Settings
from quantlab.data import Catalog, load_catalog
from quantlab.graph import run_graph
from quantlab.guardrails import Action
from quantlab.lineage import replay
from quantlab.mock_scenarios import (
    DIP_REQUEST,
    SCENARIOS,
    SCENARIOS_BY_ID,
    VIX_CHF_NARRATIVE,
    VIX_CHF_REQUEST,
    Scenario,
    match_scenario,
)
from quantlab.narrator import AI_WRITTEN_NOTE, NUMBERS_CHECKED_NOTE
from quantlab.nodes import refusal_reason
from quantlab.plan import Plan
from quantlab.registry import REGISTRY
from quantlab.validator import fill_defaults, validate_plan

from conftest import (
    ARIMA_PLAN,
    SCENARIO_1_APPROVALS,
    SCENARIO_5_APPROVALS,
    edit,
    execute,
    start_review,
)
from fakes import FakeAnthropicClient, tool_response


def test_examples_are_numbered_in_list_order() -> None:
    # The order of the sidebar buttons and of `cli demo`: the Swiss franc analysis, the
    # three blocks, then buy the dip.
    assert [(s.number, s.id) for s in SCENARIOS] == [
        (1, "chf_safe_haven_success"),
        (2, "spread_frequency_mismatch_block"),
        (3, "arima_forecast_block"),
        (4, "prompt_injection_unknown_capability"),
        (5, "buy_the_dip_tech"),
    ]
    assert all(s.label.startswith(f"{s.number} · ") for s in SCENARIOS)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
def test_scenario_end_to_end(scenario: Scenario, settings: Settings) -> None:
    state = run_graph(
        scenario.request,
        list(scenario.dataset_ids),
        "mock",
        approve=scenario.auto_approve,
        settings=settings,
    )
    assert state["status"] == scenario.expected_status, state["record"].error
    assert state["report"].codes() == sorted(scenario.expected_codes)
    if scenario.expected_status == "completed":
        # Runtime guardrails may only warn here. Which ones depends on the data (E04 on
        # the committed snapshots, where the VIX starts in 1990; E05 for example 5 on the
        # synthetic VIX); the golden records pin them on both data sets.
        assert all(i.action == Action.WARN for i in state["runtime_issues"])
        assert state["narrative"].grounding.grounded
    else:
        assert "steps" not in state  # nothing ran


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


# --- example 1: the canned plan is the planner's reading -----------------------------

# (step id, capability, inputs, params): the plan example 1's request specifies, in the
# order the live planner proposed it (run 20261019-102244-eb33f1, whose narrative mock
# mode replays): the VIX to its flags, then USD/CHF to its next-week return, then the test.
SCENARIO_1_PLAN = [
    ("s1", "load_series", {}, {"dataset_id": "VIXCLS"}),
    ("s2", "resample", {"series": "s1"}, {"freq": "W", "how": "last"}),
    ("s3", "flag_regime", {"series": "s2"},
     {"method": "expanding", "quantile": 0.90, "min_periods": 52, "direction": "above"}),
    ("s4", "load_series", {}, {"dataset_id": "DEXSZUS"}),
    ("s5", "resample", {"series": "s4"}, {"freq": "W", "how": "last"}),
    ("s6", "forward_return", {"series": "s5"}, {"method": "simple"}),
    ("s7", "hac_mean_difference", {"outcome": "s6", "flags": "s3"}, {}),
]  # fmt: skip


def test_scenario_1_plan_is_exactly_the_specified_plan() -> None:
    plan = SCENARIOS_BY_ID["chf_safe_haven_success"].plan
    steps = [(s["id"], s["capability"], s["inputs"], s["params"]) for s in plan["steps"]]
    assert steps == SCENARIO_1_PLAN
    assert plan["dataset_ids"] == ["VIXCLS", "DEXSZUS"] and not plan["cannot_fulfill"]
    # The assumptions the plan declares to the analyst: how it read "high" and "the
    # following week" (the request leaves them open), and which way USD/CHF reads.
    notes = plan["planner_notes"]
    assert "a high VIX is read as the top 10% of its past levels" in notes
    assert "(its Friday close)" in notes and "past levels only" in notes
    assert "The following week is read as weekly data" in notes
    assert "next week's simple USD/CHF return" in notes
    assert "a negative return means the franc strengthened" in notes


def test_requests_rely_on_these_registry_defaults() -> None:
    # The requests leave them to the registry (example 1 its direction, every example
    # its minimum history; example 5's canned plan omits every parameter with a
    # default), so a planner that omits them gets the canned plan.
    defaults = {
        ("resample", "how"): "last",
        ("compute_returns", "periods"): 1,
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


def test_scenario_1_validates_with_exactly_one_v10_approval(catalog: Catalog) -> None:
    raw = SCENARIOS_BY_ID["chf_safe_haven_success"].plan
    plan = fill_defaults(Plan(plan_id="p", request=VIX_CHF_REQUEST, **raw))
    report = validate_plan(plan, catalog)
    # The outcome's return method, and nothing else: the VIX level is flagged directly
    # (no returns), both series are weekly, and the threshold uses past data only.
    assert [(i.issue_id, i.action) for i in report.issues] == [("V10:s6:return_method", "APPROVE")]
    assert "method='simple'" in report.issues[0].message


def test_scenario_1_runs_end_to_end_and_replays_identically(settings: Settings) -> None:
    state = run_graph(VIX_CHF_REQUEST, ["VIXCLS", "DEXSZUS"], "mock", approve=True,
                      settings=settings)  # fmt: skip
    assert state["status"] == "completed"
    assert [a.issue_id for a in state["approvals"]] == SCENARIO_1_APPROVALS
    assert state["narrative"].grounding.grounded
    report = replay(state["run_id"], settings)
    assert report.all_match, report.message
    original = [r.output_hash for r in state["steps"]]
    assert [s.replay_hash for s in report.steps] == original and len(original) == 7


def test_scenario_1_tests_usd_chf_returns_by_vix_flags(settings: Settings) -> None:
    graph, run = start_review("chf_safe_haven_success", settings)
    state = execute(graph, run, SCENARIO_1_APPROVALS).state
    artifacts = state["artifacts"]
    # The flags come from the weekly VIX level, the outcome from USD/CHF.
    assert artifacts["s2"].meta.kind == "index_level" and artifacts["s2"].meta.frequency == "W"
    assert artifacts["s3"].meta.source_dataset_ids == ["VIXCLS"]
    assert artifacts["s6"].meta.source_dataset_ids == ["DEXSZUS"]
    test = artifacts["s7"].value
    # Every flagged week but the last, which has no next-week return yet.
    flags = artifacts["s3"].value
    assert test["horizon"] == 1 and test["n_flagged"] == int(flags.iloc[:-1].sum())
    assert artifacts["s7"].meta.source_dataset_ids == ["DEXSZUS", "VIXCLS"]
    # The test reports both groups' means: the flagged weeks' and the other weeks'.
    assert test["mean_flagged"] == pytest.approx(test["mean_normal"] + test["diff_means"])
    # The narrative states which way USD/CHF reads, and quotes only computed numbers.
    narrative = state["narrative"]
    assert (
        "DEXSZUS is quoted as USD/CHF, in Swiss francs per US dollar: a negative return "
        "means the franc strengthened against the dollar" in narrative.text
    )
    # Both means and both group sizes, as computed by the test (s7).
    facts = {fact.fact_id: fact.display for fact in narrative.facts}
    for key in ("mean_flagged", "mean_normal", "n_flagged", "n_normal"):
        assert facts[f"s7.{key}"] in narrative.text, key
    assert narrative.grounding.grounded and narrative.issues == []


def committed_snapshots(runs_dir: Path) -> Settings:
    """Settings for the committed FRED snapshots, the data example 1's narrative was
    recorded on (skips the test if the snapshots changed since)."""
    settings = Settings(anthropic_api_key=None, planner_model="test-model",
                        narrator_model="test-model", llm_mode="mock",
                        data_dir=PROJECT_ROOT / "data", runs_dir=runs_dir)  # fmt: skip
    catalog = load_catalog(settings.catalog_path)
    if any(catalog.entries[d].sha256 != h for d, h in VIX_CHF_NARRATIVE.dataset_sha256.items()):
        pytest.skip("the committed snapshots changed since example 1's narrative was recorded")
    return settings


def test_scenario_1_replays_its_recorded_narrative_on_the_committed_data(tmp_path: Path) -> None:
    settings = committed_snapshots(tmp_path / "runs")
    state = run_graph(VIX_CHF_REQUEST, ["VIXCLS", "DEXSZUS"], "mock", approve=True,
                      settings=settings)  # fmt: skip
    narrative = state["narrative"]
    # Word for word the live run's narrative, and its numbers checked again (N01).
    assert narrative.text == VIX_CHF_NARRATIVE.text and len(narrative.text.split()) == 95
    assert narrative.source == "mock" and narrative.model is None
    assert narrative.grounding.grounded and narrative.issues == [] and narrative.withheld is None
    assert narrative.note == f"{AI_WRITTEN_NOTE} {NUMBERS_CHECKED_NOTE}"


def test_an_edited_scenario_1_plan_gets_the_template(tmp_path: Path) -> None:
    graph, run = start_review("chf_safe_haven_success", committed_snapshots(tmp_path / "runs"))
    run = edit(graph, run, {"s3": {"quantile": 0.95}})  # other numbers than recorded
    narrative = execute(graph, run, SCENARIO_1_APPROVALS).state["narrative"]
    assert narrative.source == "template" and narrative.issues == []


def test_scenario_1_on_other_data_gets_the_template(settings: Settings) -> None:
    # The synthetic test data: not the files the narrative was recorded on.
    state = run_graph(VIX_CHF_REQUEST, ["VIXCLS", "DEXSZUS"], "mock", approve=True,
                      settings=settings)  # fmt: skip
    assert state["narrative"].source == "template" and state["narrative"].issues == []


def test_a_recorded_narrative_that_fails_n01_is_withheld(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A recorded text with one wrong number is treated like an LLM narrative that
    # fails the check: withheld, N01 raised, the template shown.
    wrong = dataclasses.replace(
        VIX_CHF_NARRATIVE, text=VIX_CHF_NARRATIVE.text.replace("0.5069", "0.4069")
    )
    scenario = dataclasses.replace(SCENARIOS_BY_ID["chf_safe_haven_success"], narrative=wrong)
    monkeypatch.setattr("quantlab.narrator.match_scenario", lambda request: scenario)
    state = run_graph(VIX_CHF_REQUEST, ["VIXCLS", "DEXSZUS"], "mock", approve=True,
                      settings=committed_snapshots(tmp_path / "runs"))  # fmt: skip
    narrative = state["narrative"]
    assert narrative.source == "template" and narrative.withheld.source == "mock"
    assert narrative.withheld.grounding.ungrounded == ["0.4069"]
    assert [i.code for i in narrative.issues] == ["N01"]
    assert narrative.error == "The recorded narrative was withheld by N01 UNGROUNDED_NARRATIVE."


def test_scenario_1_demo_edit_full_history_threshold_needs_the_look_ahead_approval(
    settings: Settings,
) -> None:
    graph, run = start_review("chf_safe_haven_success", settings)
    run = edit(graph, run, {"s3": {"method": "full_sample"}})
    report = run.state["report"]
    # The VIX threshold from its entire history: V09, an approval (not a block), which
    # also says the significance test is invalidated; the return-method approval stays.
    assert [(i.issue_id, i.action) for i in report.issues] == [
        ("V09:s3", "APPROVE"),
        ("V10:s6:return_method", "APPROVE"),
    ]
    v09 = report.issues[0]
    assert v09.name == "LOOK_AHEAD"
    assert "It also invalidates the significance test in s7" in v09.message
    refused = execute(graph, run, ["V10:s6:return_method"])  # not without the analyst's decision
    assert refused.paused_for["error"] == "Approval required for: V09:s3"
    run = execute(graph, refused, ["V09:s3", "V10:s6:return_method"])
    assert run.state["status"] == "completed"


def test_scenario_1_demo_edit_dataset_to_a_yield_is_blocked(settings: Settings) -> None:
    graph, run = start_review("chf_safe_haven_success", settings)
    run = edit(graph, run, {"s4": {"dataset_id": "DGS10"}})
    report = run.state["report"]
    # A legal edit (a parameter value), which the re-validation blocks: a forward
    # "return" needs a price level, and a yield is a rate. (V07 RETURNS_ON_RATES needs
    # a compute_returns step: the same edit on example 5 shows it.)
    assert run.state["plan"].steps[3].params["dataset_id"] == "DGS10"
    assert [(i.issue_id, i.name) for i in report.blocks] == [("V04:s6:series", "TYPE_MISMATCH")]
    refused = execute(graph, run, SCENARIO_1_APPROVALS)
    assert refused.paused_for["error"].startswith("The plan has blocking issues")


def test_scenario_1_demo_edit_quantile_out_of_range_is_blocked(settings: Settings) -> None:
    graph, run = start_review("chf_safe_haven_success", settings)
    run = edit(graph, run, {"s3": {"quantile": 1.5}})
    (v03,) = run.state["report"].blocks
    assert (v03.code, v03.name, v03.step_id) == ("V03", "INVALID_PARAM", "s3")
    assert "'quantile'=1.5 is outside [0.01, 0.99]" in v03.message
    refused = execute(graph, run, SCENARIO_1_APPROVALS)
    assert refused.paused_for["error"].startswith("The plan has blocking issues")


# --- example 5, buy the dip: the bottom tail of past weekly returns ---------------

# (step id, capability, inputs, params): the plan example 5's request specifies. Only
# the parameters the request states or the registry requires; the defaults fill the rest.
DIP_PLAN = [
    ("s1", "load_series", {}, {"dataset_id": "NASDAQCOM"}),
    ("s2", "resample", {"series": "s1"}, {"freq": "W"}),
    ("s3", "compute_returns", {"series": "s2"}, {"method": "simple"}),
    ("s4", "flag_regime", {"series": "s3"}, {"quantile": 0.10, "direction": "below"}),
    ("s5", "forward_return", {"series": "s2"}, {"method": "simple"}),
    ("s6", "hac_mean_difference", {"outcome": "s5", "flags": "s4"}, {}),
]  # fmt: skip


def test_scenario_5_plan_is_exactly_the_specified_plan() -> None:
    plan = SCENARIOS_BY_ID["buy_the_dip_tech"].plan
    steps = [(s["id"], s["capability"], s["inputs"], s["params"]) for s in plan["steps"]]
    assert steps == DIP_PLAN
    assert plan["dataset_ids"] == ["NASDAQCOM"] and not plan["cannot_fulfill"]
    notes = plan["planner_notes"]
    assert "below the 10th percentile of past weekly returns only" in notes
    assert "simple weekly returns" in notes and "default expanding threshold" in notes
    # The plan the analyst reviews has every default filled in: past data only.
    filled = fill_defaults(Plan(plan_id="p", request=DIP_REQUEST, **plan))
    assert filled.steps[1].params == {"freq": "W", "how": "last"}
    assert filled.steps[2].params == {"method": "simple", "periods": 1}
    assert filled.steps[3].params == {
        "quantile": 0.10, "direction": "below", "method": "expanding", "min_periods": 52,
    }  # fmt: skip


def test_scenario_5_validates_with_exactly_two_v10_approvals(catalog: Catalog) -> None:
    raw = SCENARIOS_BY_ID["buy_the_dip_tech"].plan
    plan = fill_defaults(Plan(plan_id="p", request=DIP_REQUEST, **raw))
    report = validate_plan(plan, catalog)
    assert [(i.issue_id, i.action) for i in report.issues] == [
        ("V10:s3:return_method", "APPROVE"),  # simple returns, the ones flagged
        ("V10:s5:return_method", "APPROVE"),  # simple returns for the outcome
    ]  # nothing else: no WARN, no BLOCK, no look-ahead (an expanding threshold)


def test_scenario_5_runs_end_to_end_and_replays_identically(settings: Settings) -> None:
    state = run_graph(DIP_REQUEST, ["NASDAQCOM"], "mock", approve=True, settings=settings)
    assert state["status"] == "completed" and not state["runtime_issues"]
    assert [a.issue_id for a in state["approvals"]] == SCENARIO_5_APPROVALS
    assert state["narrative"].grounding.grounded
    report = replay(state["run_id"], settings)
    assert report.all_match, report.message
    original = [r.output_hash for r in state["steps"]]
    assert [s.replay_hash for s in report.steps] == original and len(original) == 6


def test_scenario_5_flags_the_biggest_drops_from_past_data_only(settings: Settings) -> None:
    graph, run = start_review("buy_the_dip_tech", settings)
    state = execute(graph, run, SCENARIO_5_APPROVALS).state
    artifacts = state["artifacts"]
    returns, flags = artifacts["s3"].value, artifacts["s4"].value
    # Each dip is below the 10% quantile of the returns BEFORE it (no look-ahead).
    threshold = returns.shift(1).expanding(min_periods=52).quantile(0.10)
    assert flags.any() and (returns[flags.index][flags] < threshold[flags.index][flags]).all()
    assert (returns[flags.index][flags] < 0).all()  # every dip is a drop
    # The narrative answers the two-sided question ("differ"), then says which mean was
    # higher with both means, in grounded numbers.
    narrative = state["narrative"]
    test = artifacts["s6"].value
    assert "The test asks whether next-week returns after flagged weeks differ" in narrative.text
    higher = "flagged weeks" if test["diff_means"] > 0 else "the other weeks"
    assert f"The mean was higher after {higher}:" in narrative.text
    facts = {fact.fact_id: fact.display for fact in narrative.facts}
    for key in ("mean_flagged", "mean_normal", "n_flagged", "n_normal", "n_episodes_flagged"):
        assert facts[f"s6.{key}"] in narrative.text, key
    assert narrative.grounding.grounded and narrative.issues == []


def test_scenario_5_demo_edit_dataset_to_a_yield_blocks_returns_on_rates(
    settings: Settings,
) -> None:
    graph, run = start_review("buy_the_dip_tech", settings)
    run = edit(graph, run, {"s1": {"dataset_id": "DGS10"}})
    # A legal edit, which the re-validation blocks: "returns" of a yield are
    # meaningless (V07), and a forward return needs a price level (V04).
    assert run.state["plan"].steps[0].params["dataset_id"] == "DGS10"
    assert [i.issue_id for i in run.state["report"].blocks] == ["V07:s3", "V04:s5:series"]
    refused = execute(graph, run, SCENARIO_5_APPROVALS)
    assert refused.paused_for["error"].startswith("The plan has blocking issues")


def test_scenario_3_refusal_lists_what_the_registry_can_do(settings: Settings) -> None:
    graph, run = start_review("arima_forecast_block", settings)
    reason = run.state["plan"].refusal_reason
    assert reason.startswith("No registered capability fits a model or makes a forecast")
    assert "ARIMA forecast of the S&P 500" in reason
    assert all(name in reason for name in REGISTRY)  # the list comes from the registry
    assert run.state["status"] == "blocked" and "steps" not in run.state


def test_scenario_3_wall_holds_even_if_the_planner_misbehaves(settings: Settings) -> None:
    """A planner that emits an 'arima' step instead of refusing is still blocked."""
    request = SCENARIOS_BY_ID["arima_forecast_block"].request
    client = FakeAnthropicClient(tool_response(ARIMA_PLAN))
    state = run_graph(request, ["NASDAQCOM"], "live", settings=settings, client=client)
    assert state["report"].codes() == ["V02"]
    assert state["status"] == "blocked" and "steps" not in state


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
    state = run_graph(scenario.request, list(scenario.dataset_ids), "mock", settings=settings)

    # A well-formed plan (no V01) of one step that follows the injection, rejected
    # whole by V02.
    assert [s.capability for s in state["plan"].steps] == ["run_python"]
    assert state["plan"].steps[0].params == {"code": "import os; os.system('ls')"}
    assert [(i.issue_id, i.action) for i in state["report"].issues] == [("V02:s1", "BLOCK")]
    assert state["report"].is_rejected
    # Blocked before approval: the executor is never reached, no step ever runs.
    assert state["status"] == "blocked" and "steps" not in state and not state.get("approvals")
    assert "run_python" not in REGISTRY
    assert shell_commands == []  # os.system / os.popen never called
    # The only processes are the run record's fixed, read-only git queries (commit
    # hash and dirty flag); nothing from the request ever reaches a process.
    assert processes and all(args[0] == "git" for args in processes)
    assert not any("ls" in args or "os.system" in " ".join(args) for args in processes)


def test_scenario_2_frequency_mismatch_blocks_before_anything_runs(settings: Settings) -> None:
    graph, run = start_review("spread_frequency_mismatch_block", settings)
    report = run.state["report"]
    # The frequency mismatch, and the look-ahead approval of the full-history threshold.
    assert [(i.issue_id, i.action) for i in report.issues] == [
        ("V08:s4", "BLOCK"),
        ("V09:s5", "APPROVE"),
    ]
    v08 = report.blocks[0]
    assert "'a' is weekly, 'b' is daily" in v08.message
    assert "Resample DGS2 to weekly" in v08.suggestion
    assert run.state["status"] == "blocked" and refusal_reason(run.state)
    refused = execute(graph, run, ["V09:s5"])
    assert refused.paused_for["error"] == "The plan has blocking issues: V08:s4"


def test_scenario_2_asks_for_the_bottom_tail_of_the_spread() -> None:
    flags = SCENARIOS_BY_ID["spread_frequency_mismatch_block"].plan["steps"][4]
    assert (flags["capability"], flags["inputs"]) == ("flag_regime", {"series": "s4"})
    assert flags["params"] == {"method": "full_sample", "quantile": 0.10, "min_periods": 52,
                               "direction": "below"}  # fmt: skip


def test_scenario_2_runs_once_both_yields_are_daily(settings: Settings) -> None:
    # The fix the app allows (steps are locked): the 10-year kept daily, like the 2-year.
    graph, run = start_review("spread_frequency_mismatch_block", settings)
    run = edit(graph, run, {"s2": {"freq": "D"}})
    assert [i.issue_id for i in run.state["report"].issues] == ["V09:s5"]  # no more V08
    state = execute(graph, run, ["V09:s5"]).state
    assert state["status"] == "completed"
    spread = state["artifacts"]["s4"].value
    flags = state["artifacts"]["s5"].value
    # The bottom tail of the whole history: below the 10% quantile, about 10% of the days.
    assert flags.any() and (spread[flags] < spread.quantile(0.10)).all()
    assert 0.09 < flags.mean() < 0.11
