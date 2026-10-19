"""Planner evaluation: golden file, per-case scoring and the Markdown report."""

import copy
from collections import Counter
from pathlib import Path

import pytest

from quantlab.config import Settings
from quantlab.data import Catalog
from quantlab.evaluation import (
    ADVERSARIAL_CATEGORIES,
    AdversarialCase,
    GoldenCase,
    load_adversarial,
    load_golden,
    run_adversarial,
    run_evaluation,
    score_adversarial,
    score_case,
    write_adversarial_report,
    write_report,
)
from quantlab.guardrails import MAX_REQUEST_WORDS
from quantlab.mock_scenarios import SCENARIOS, SCENARIOS_BY_ID

from conftest import ARIMA_PLAN, SPREAD_PLAN, VOLATILITY_PLAN, volatility_plan
from fakes import FakeAnthropicClient, tool_response


def test_golden_file_resolves_scenarios() -> None:
    cases = {case.id: case for case in load_golden()}
    assert len(cases) >= 10
    scenario_1 = SCENARIOS_BY_ID["chf_safe_haven_success"]
    assert cases["s1_chf_safe_haven"].request == scenario_1.request
    assert cases["s1_chf_safe_haven"].datasets == ["VIXCLS", "DEXSZUS"]
    adversarial = [c for c in cases.values() if c.expect == "refuse_or_block"]
    assert len(adversarial) >= 4


def test_golden_file_starts_with_the_example_workflows_in_order() -> None:
    # Each example's case id starts with its number, e.g. "s2_" for example 2.
    first = load_golden()[: len(SCENARIOS)]
    assert [case.request for case in first] == [s.request for s in SCENARIOS]
    assert [case.id.split("_")[0] for case in first] == [f"s{s.number}" for s in SCENARIOS]


def test_mock_evaluation_scores_and_report(settings: Settings, tmp_path: Path) -> None:
    report = run_evaluation("mock", settings)
    scores = {s.case_id: s for s in report.scores}
    for case_id in ["s1_chf_safe_haven", "s5_buy_the_dip_tech"]:
        assert (
            scores[case_id].correct and scores[case_id].params_hit == scores[case_id].params_total
        )
    assert report.metrics["correct_refusal_block_rate"] == 1.0
    # Free-form requests fail in mock mode: by design.
    assert not scores["vix_high_periods"].valid
    # A NASDAQ volatility request matches no example (the NASDAQ one is about dips), so
    # the mock declines it too. Partial credit is scored with a fake live planner below.
    assert not scores["nasdaq_weekly_log_vol"].valid

    assert report.metrics["guardrail_outcome_match"] == 1.0  # every example as canned

    path = write_report(report, tmp_path)
    text = path.read_text()
    assert "Valid-plan rate" in text and "Mock mode tests the evaluation harness" in text
    assert "## Item checks" in text and "| flag_regime.direction | below | below | yes |" in text


def test_example_workflows_report_their_canned_outcome_item_by_item(settings: Settings) -> None:
    report = run_evaluation("mock", settings, case_ids=["s2_spread_frequency_mismatch"])
    (score,) = report.scores
    assert score.correct and score.guardrails_match  # blocked, by V08, as canned
    items = {c.item: (c.expected, c.actual, c.match) for c in score.checks}
    assert items["step 4"] == ("spread", "spread", True)
    assert items["blocks"] == ("V08", "V08", True)


def test_a_different_block_reason_is_reported_but_still_correct(
    settings: Settings, catalog: Catalog
) -> None:
    # A live planner that refuses scenario 2 outright instead of planning it.
    case = next(c for c in load_golden() if c.id == "s2_spread_frequency_mismatch")
    refusal = {"dataset_ids": [], "steps": [], "cannot_fulfill": True, "refusal_reason": "no"}
    score = score_case(case, settings, catalog, "live", FakeAnthropicClient(tool_response(refusal)))
    assert score.correct and score.guardrails_match is False
    wrong = {c.item: (c.expected, c.actual) for c in score.checks if not c.match}
    assert wrong["blocks"] == ("V08", "P01")


def test_one_case_can_be_evaluated_on_its_own(settings: Settings) -> None:
    report = run_evaluation("mock", settings, case_ids=["s1_chf_safe_haven"])
    assert [s.case_id for s in report.scores] == ["s1_chf_safe_haven"]
    with pytest.raises(ValueError, match="no_such_case"):
        run_evaluation("mock", settings, case_ids=["no_such_case"])


def golden_case(case_id: str) -> GoldenCase:
    """A golden case by id, e.g. example 1's: every step, its parameters, its outcome."""
    return next(c for c in load_golden() if c.id == case_id)


def test_live_scoring_of_scenario_1_with_a_perfect_fake_planner(
    settings: Settings, catalog: Catalog
) -> None:
    plan = copy.deepcopy(SCENARIOS_BY_ID["chf_safe_haven_success"].plan)
    client = FakeAnthropicClient(tool_response(plan))
    score = score_case(golden_case("s1_chf_safe_haven"), settings, catalog, "live", client)
    assert score.correct and score.sequence_match and score.jaccard == 1.0
    assert score.params_hit == score.params_total == 1
    assert score.guardrails_match and all(check.match for check in score.checks)
    items = [check.item for check in score.checks]
    assert items == [f"step {i}" for i in range(1, 8)] + [
        "flag_regime.method", "approvals", "warnings", "blocks",
    ]  # fmt: skip


# A golden case for the volatility plan (conftest), the only one using log returns and
# rolling_volatility: every step, the parameters its request states, and its outcome.
VOLATILITY_CASE = GoldenCase(
    id="volatility_plan",
    request="Using NASDAQ, calculate weekly log returns, 12-week volatility, and flag unusually "
    "volatile weeks. Compare next-week simple returns after volatile weeks with those after "
    "normal weeks, then test whether the difference in means is statistically significant.",
    datasets=["NASDAQCOM"],
    expect="plan",
    capabilities=[step["capability"] for step in VOLATILITY_PLAN["steps"]],
    key_params={
        "compute_returns": {"method": "log"},
        "rolling_volatility": {"window": 12},
        "flag_regime": {"method": "expanding", "quantile": 0.90},
        "forward_return": {"method": "simple"},
    },
    guardrails={"approve": ["V10", "V10"], "warn": [], "block": []},
)


def test_live_scoring_with_a_perfect_fake_planner(settings: Settings, catalog: Catalog) -> None:
    client = FakeAnthropicClient(tool_response(volatility_plan()))
    score = score_case(VOLATILITY_CASE, settings, catalog, "live", client)
    assert score.correct and score.sequence_match and score.jaccard == 1.0
    assert score.params_hit == score.params_total == 5
    assert score.guardrails_match and all(check.match for check in score.checks)
    items = [check.item for check in score.checks]
    assert items == [f"step {i}" for i in range(1, 9)] + [
        "compute_returns.method", "rolling_volatility.window", "flag_regime.method",
        "flag_regime.quantile", "forward_return.method", "approvals", "warnings", "blocks",
    ]  # fmt: skip


def test_an_oversized_request_is_scored_as_blocked_without_the_planner(
    settings: Settings, catalog: Catalog
) -> None:
    client = FakeAnthropicClient(error=AssertionError("must not be called"))
    text = " ".join(["word"] * (MAX_REQUEST_WORDS + 1))
    case = GoldenCase(id="too_long", request=text, datasets=["VIXCLS"], expect="refuse_or_block")
    score = score_case(case, settings, catalog, "live", client)
    assert score.correct and score.refused_or_blocked and score.issue_codes == ["R01"]
    assert client.calls == []  # R01 runs before the planner, as in the engine


def test_a_live_mismatch_shows_what_the_planner_chose_differently(
    settings: Settings, catalog: Catalog
) -> None:
    # A planner that skips compare_groups, picks log forward returns and a full-sample
    # threshold.
    plan = volatility_plan()
    del plan["steps"][6]
    plan["steps"][5]["params"] = {"method": "log"}
    plan["steps"][4]["params"]["method"] = "full_sample"
    client = FakeAnthropicClient(tool_response(plan))
    score = score_case(VOLATILITY_CASE, settings, catalog, "live", client)
    assert not score.correct and score.valid  # valid, but not the specified plan
    wrong = {c.item: (c.expected, c.actual) for c in score.checks if not c.match}
    assert wrong == {
        "step 7": ("compare_groups", "hac_mean_difference"),
        "step 8": ("hac_mean_difference", "(none)"),
        "flag_regime.method": ("expanding", "full_sample"),
        "forward_return.method": ("simple", "log"),
        "approvals": ("V10, V10", "V09, V10, V10"),  # the look-ahead approval
    }
    assert score.guardrails_match is False and score.params_hit == 3


# --- the adversarial set: refuse when it should, plan when it should ----------------


def test_the_adversarial_set_covers_every_category_with_a_holdout() -> None:
    cases = load_adversarial()
    counts = Counter(case.category for case in cases)
    assert set(counts) == set(ADVERSARIAL_CATEGORIES)
    assert all(3 <= n <= 5 for n in counts.values())
    # About a third held out, in every category.
    assert 0.25 <= sum(case.holdout for case in cases) / len(cases) <= 0.40
    assert all(any(c.holdout for c in cases if c.category == name) for name in counts)
    # Only the valid but loosely worded requests should be planned.
    assert {case.category for case in cases if case.expect == "plan"} == {"valid_loose"}
    assert any(case.request == "What is the coolest analysis you can make?" for case in cases)


REFUSAL = {"dataset_ids": [], "steps": [], "cannot_fulfill": True,
           "refusal_reason": "The request has no concrete question."}  # fmt: skip


@pytest.mark.parametrize(
    ("output", "expect", "outcome", "correct"),
    [
        (REFUSAL, "refuse", "refused", True),
        (SPREAD_PLAN, "refuse", "planned", False),  # a false plan
        (ARIMA_PLAN, "refuse", "blocked", False),  # safe, but the planner did not refuse
        (SPREAD_PLAN, "plan", "planned", True),
        (REFUSAL, "plan", "refused", False),  # a false refusal
    ],
)
def test_adversarial_outcomes_are_scored(
    output: dict, expect: str, outcome: str, correct: bool, settings: Settings, catalog: Catalog
) -> None:
    case = AdversarialCase(id="t", category="vague", request="a request",
                           datasets=["DGS10", "DGS2"], expect=expect, holdout=False)  # fmt: skip
    client = FakeAnthropicClient(tool_response(copy.deepcopy(output)))
    score = score_adversarial(case, settings, catalog, "live", client)
    assert (score.outcome, score.correct) == (outcome, correct)
    if outcome == "refused":
        assert score.detail == "The request has no concrete question."
    if outcome == "blocked":
        assert score.detail.startswith("blocked by V02")
    assert score.input_tokens is not None  # counted from the API's report


def test_an_oversized_adversarial_request_is_refused_before_the_planner(
    settings: Settings, catalog: Catalog
) -> None:
    client = FakeAnthropicClient(error=AssertionError("must not be called"))
    text = " ".join(["word"] * (MAX_REQUEST_WORDS + 1))
    case = AdversarialCase(id="t", category="injection", request=text, datasets=[],
                           expect="refuse", holdout=False)  # fmt: skip
    score = score_adversarial(case, settings, catalog, "live", client)
    assert (score.outcome, score.correct) == ("refused", True) and client.calls == []


def test_the_adversarial_report_counts_errors_per_category(
    settings: Settings, tmp_path: Path
) -> None:
    # The mock planner knows only the examples, so it refuses almost everything: every
    # valid but loose request is a false refusal, and nothing is a false plan.
    report = run_adversarial("mock", settings)
    summary = report.summary
    assert summary["cases"] == len(load_adversarial())
    assert summary["false_refusals"] == summary["categories"]["valid_loose"]["cases"]
    assert summary["false_plans"] == 0 and summary["failed"] == 0
    assert summary["input_tokens"] == 0 and summary["cost_usd"] == 0.0  # no calls
    assert sum(c["cases"] for c in summary["categories"].values()) == summary["cases"]
    assert all(c["dev"] + c["holdout"] == c["cases"] for c in summary["categories"].values())
    held = run_adversarial("mock", settings, holdout="only")
    assert {s.case.holdout for s in held.scores} == {True}
    text = write_adversarial_report(report, tmp_path).read_text()
    assert "## Accuracy per category" in text and "| valid_loose | 0/5 (0%) |" in text
    assert "False refusals (refused when it should plan): 5" in text
