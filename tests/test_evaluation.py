"""Planner evaluation: golden file, per-case scoring and the Markdown report."""

import copy
from pathlib import Path

from quantlab.config import Settings
from quantlab.data import Catalog
from quantlab.evaluation import load_golden, run_evaluation, score_case, write_report
from quantlab.scenarios import SCENARIOS_BY_ID

from fakes import FakeAnthropicClient, tool_response


def test_golden_file_resolves_scenarios() -> None:
    cases = {case.id: case for case in load_golden()}
    assert len(cases) >= 12
    assert cases["A_volatility_regimes"].request == SCENARIOS_BY_ID["A"].request
    adversarial = [c for c in cases.values() if c.expect == "refuse_or_block"]
    assert len(adversarial) >= 4


def test_mock_evaluation_scores_and_report(settings: Settings, tmp_path: Path) -> None:
    report = run_evaluation("mock", settings)
    scores = {s.case_id: s for s in report.scores}
    for case_id in ["A_volatility_regimes", "B_full_sample_threshold", "F_rates_spread"]:
        assert (
            scores[case_id].correct and scores[case_id].params_hit == scores[case_id].params_total
        )
    assert report.metrics["correct_refusal_block_rate"] == 1.0
    # The mock only knows the demo scenarios, so free-form requests fail: by design.
    assert not scores["vix_summary"].valid
    # The A-like variation gets A's canned plan: right shape, wrong parameters.
    variation = scores["nasdaq_variation_mann_whitney"]
    assert variation.jaccard > 0.8 and variation.params_hit < variation.params_total

    path = write_report(report, tmp_path)
    text = path.read_text()
    assert "Valid-plan rate" in text and "Mock mode tests the evaluation harness" in text


def test_live_scoring_with_a_perfect_fake_planner(settings: Settings, catalog: Catalog) -> None:
    case = next(c for c in load_golden() if c.id == "F_rates_spread")
    client = FakeAnthropicClient(tool_response(copy.deepcopy(SCENARIOS_BY_ID["F"].plan)))
    score = score_case(case, settings, catalog, "live", client)
    assert score.correct and score.sequence_match and score.jaccard == 1.0
    assert score.params_hit == score.params_total == 2
