"""Planner evaluation: golden file, per-case scoring and the Markdown report."""

import copy
from pathlib import Path

import pytest

from quantlab.config import Settings
from quantlab.data import Catalog
from quantlab.evaluation import GoldenCase, load_golden, run_evaluation, score_case, write_report
from quantlab.mock_scenarios import SCENARIOS_BY_ID

from fakes import FakeAnthropicClient, tool_response


def test_golden_file_resolves_scenarios() -> None:
    cases = {case.id: case for case in load_golden()}
    assert len(cases) >= 12
    scenario_1 = SCENARIOS_BY_ID["volatility_flag_success"]
    assert cases["s1_volatility_flag_success"].request == scenario_1.request
    adversarial = [c for c in cases.values() if c.expect == "refuse_or_block"]
    assert len(adversarial) >= 4


def test_mock_evaluation_scores_and_report(settings: Settings, tmp_path: Path) -> None:
    report = run_evaluation("mock", settings)
    scores = {s.case_id: s for s in report.scores}
    for case_id in ["s1_volatility_flag_success", "s2_volatility_flag_lookahead"]:
        assert (
            scores[case_id].correct and scores[case_id].params_hit == scores[case_id].params_total
        )
    assert report.metrics["correct_refusal_block_rate"] == 1.0
    # The canned requests have mock plans, scored like any other plan...
    for case_id in ["nasdaq_volatility_peak_since_2020", "vix_extremes", "sp500_returns_2022"]:
        assert scores[case_id].correct, case_id
        assert scores[case_id].params_hit == scores[case_id].params_total, case_id
    # ...but free-form requests fail in mock mode: by design.
    assert not scores["vix_summary"].valid
    # A scenario-1-like variation gets scenario 1's canned plan: every expected step
    # and as many more (Jaccard 0.5), but not the requested window.
    variation = scores["nasdaq_weekly_log_vol"]
    assert variation.jaccard == 0.5 and variation.params_hit < variation.params_total

    assert report.metrics["guardrail_outcome_match"] == 1.0  # the flagship's two approvals

    path = write_report(report, tmp_path)
    text = path.read_text()
    assert "Valid-plan rate" in text and "Mock mode tests the evaluation harness" in text
    assert "## Item checks" in text and "| forward_return.method | simple | simple | yes |" in text


def test_one_case_can_be_evaluated_on_its_own(settings: Settings) -> None:
    report = run_evaluation("mock", settings, case_ids=["s1_volatility_flag_success"])
    assert [s.case_id for s in report.scores] == ["s1_volatility_flag_success"]
    with pytest.raises(ValueError, match="no_such_case"):
        run_evaluation("mock", settings, case_ids=["no_such_case"])


def flagship_case() -> GoldenCase:
    """The golden flagship case: every step, the parameters it states, the guardrail outcome."""
    return next(c for c in load_golden() if c.id == "s1_volatility_flag_success")


def test_live_scoring_with_a_perfect_fake_planner(settings: Settings, catalog: Catalog) -> None:
    plan = copy.deepcopy(SCENARIOS_BY_ID["volatility_flag_success"].plan)
    client = FakeAnthropicClient(tool_response(plan))
    score = score_case(flagship_case(), settings, catalog, "live", client)
    assert score.correct and score.sequence_match and score.jaccard == 1.0
    assert score.params_hit == score.params_total == 6
    assert score.guardrails_match and all(check.match for check in score.checks)
    items = [check.item for check in score.checks]
    assert items == [f"step {i}" for i in range(1, 9)] + [
        "compute_returns.method", "rolling_volatility.window", "flag_regime.method",
        "flag_regime.quantile", "forward_return.horizon", "forward_return.method",
        "approvals", "warnings", "blocks",
    ]  # fmt: skip


def test_a_live_mismatch_shows_what_the_planner_chose_differently(
    settings: Settings, catalog: Catalog
) -> None:
    # A planner that skips compare_groups and picks 4-week log forward returns.
    plan = copy.deepcopy(SCENARIOS_BY_ID["volatility_flag_success"].plan)
    del plan["steps"][6]
    plan["steps"][5]["params"] = {"horizon": 4, "method": "log"}
    client = FakeAnthropicClient(tool_response(plan))
    score = score_case(flagship_case(), settings, catalog, "live", client)
    assert not score.correct and score.valid  # valid, but not the specified plan
    wrong = {c.item: (c.expected, c.actual) for c in score.checks if not c.match}
    assert wrong == {
        "step 7": ("compare_groups", "hac_mean_difference"),
        "step 8": ("hac_mean_difference", "(none)"),
        "forward_return.horizon": ("1", "4"),
        "forward_return.method": ("simple", "log"),
        "warnings": ("none", "G11"),  # the 4-week overlap
    }
    assert score.guardrails_match is False and score.params_hit == 4
