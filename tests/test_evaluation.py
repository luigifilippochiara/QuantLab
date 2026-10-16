"""Planner evaluation: golden file, per-case scoring and the Markdown report."""

import copy
from pathlib import Path

from quantlab.config import Settings
from quantlab.data import Catalog
from quantlab.evaluation import load_golden, run_evaluation, score_case, write_report
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
    # A scenario-1-like variation gets scenario 1's canned plan: close in shape,
    # wrong parameters.
    variation = scores["nasdaq_weekly_log_vol"]
    assert variation.jaccard == 0.8 and variation.params_hit < variation.params_total

    path = write_report(report, tmp_path)
    text = path.read_text()
    assert "Valid-plan rate" in text and "Mock mode tests the evaluation harness" in text


def test_live_scoring_with_a_perfect_fake_planner(settings: Settings, catalog: Catalog) -> None:
    case = next(c for c in load_golden() if c.id == "s1_volatility_flag_success")
    plan = copy.deepcopy(SCENARIOS_BY_ID["volatility_flag_success"].plan)
    score = score_case(case, settings, catalog, "live", FakeAnthropicClient(tool_response(plan)))
    assert score.correct and score.sequence_match and score.jaccard == 1.0
    assert score.params_hit == score.params_total == 5
