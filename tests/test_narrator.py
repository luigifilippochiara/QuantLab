"""Narrator: facts sheet, formatting, numeric grounding check, template and LLM paths."""

import dataclasses

import pytest

from quantlab.config import Settings
from quantlab.data import Catalog
from quantlab.executor import execute_plan
from quantlab.guardrails import validate_plan
from quantlab.narrator import (
    Fact,
    build_facts_sheet,
    check_grounding,
    format_number,
    narrate,
    template_narrative,
)
from quantlab.plan import Plan
from quantlab.types import Artifact, ArtifactMeta, DataType, Frequency

from conftest import SPREAD_PLAN, as_plan, raw_plan
from fakes import FakeAnthropicClient, FakeOpenAIClient, openai_text_response, text_response

PLANS = {
    "scenario_1": raw_plan("volatility_flag_success"),
    "lookahead": raw_plan("vix_full_sample_lookahead"),
    "spread": SPREAD_PLAN,
}


def run_scenario(name: str, catalog: Catalog):
    """Execute a test plan on synthetic data; return plan, result and narrator issues."""
    plan = as_plan(PLANS[name])
    report = validate_plan(plan, catalog)
    result = execute_plan(plan, catalog, {i.issue_id for i in report.approvals_required})
    issues = report.warnings + report.approvals_required + result.issues
    return plan, result, issues


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.001234567, "0.001235"),
        (0.000563124, "0.0005631"),
        (27204.38, "27,204"),
        (0.9145813, "0.9146"),
        (-2.34, "-2.340"),
        (2832, "2,832"),
        (0.0, "0"),
        ("2026-10-13", "2026-10-13"),
    ],
)
def test_format_number(value: float | int | str, expected: str) -> None:
    assert format_number(value) == expected


def test_facts_sheet_covers_tables_flags_and_series(catalog: Catalog) -> None:
    plan, result, _ = run_scenario("spread", catalog)
    facts = {f.fact_id: f for f in build_facts_sheet(plan, result.artifacts)}
    for fact_id in ["s1.start", "s1.end", "s3.last_value", "s4.last_value", "s4.n_obs"]:
        assert fact_id in facts
    assert facts["s3.last_value"].value == result.artifacts["s3"].value.iloc[-1]

    plan, result, _ = run_scenario("scenario_1", catalog)
    facts = {f.fact_id: f for f in build_facts_sheet(plan, result.artifacts)}
    for fact_id in ["s4.last_value", "s5.n_flagged", "s5.share_flagged", "s5.n_episodes",
                    "s7.mean_flagged", "s7.diff_means", "s7.n_episodes_flagged"]:  # fmt: skip
        assert fact_id in facts
    assert facts["s5.n_flagged"].value == int(result.artifacts["s5"].value.sum())
    # Every number of the significance test is a fact, so every one can be checked.
    test = result.artifacts["s8"].value
    for key, value in test.items():
        assert facts[f"s8.{key}"].value == value


# --- grounding ---------------------------------------------------------------------

FACTS = [
    Fact(fact_id="s3.last_value", label="vol", value=0.138642, display="0.1386"),
    Fact(fact_id="s3.mean", label="mean", value=0.000563124, display="0.0005631"),
    Fact(fact_id="s4.n_flagged", label="n", value=2832, display="2,832"),
    Fact(fact_id="s1.start", label="start", value="1971-02-05", display="1971-02-05"),
    Fact(fact_id="s1.end", label="end", value="2026-10-13", display="2026-10-13"),
    Fact(fact_id="s5.end", label="end", value="2021-11-19", display="2021-11-19"),
]
PLAN = Plan(
    plan_id="p",
    request="r",
    steps=[
        {"id": "s4", "capability": "flag_regime", "params": {"quantile": 0.9, "min_periods": 252}}
    ],
)


@pytest.mark.parametrize(
    "sentence",
    [
        "Volatility is 0.1386 and the mean return is 0.0005631.",  # exactly as displayed
        "Volatility is 13.86% today.",  # percent form
        "Volatility is about 14% and 0.14.",  # coarser but correct rounding
        "The average daily return is 0.056%.",  # percent form of a small number
        "There were 2,832 flagged days, and 2832 is a lot.",  # thousands separator
        "Days above the 90th percentile after 252 days of history.",  # plan parameters
        "Data from 1971-02-05 to 2026-10-13, i.e. since 1971.",  # fact dates; year in range
        "It peaked on 2021-11-19, on 19 November 2021, on Nov. 19, 2021.",  # a fact date
        "It peaked in November 2021.",  # a month containing a fact date
        "The market 2021 rally and a decline 2020.",  # not dates: years in range
        "Annualised with 52 weeks per year.",  # annualisation factor
        "Steps s3 and R04 and DGS10 are named, not numbers.",  # ids are skipped
    ],
)
def test_grounding_accepts_correct_numbers(sentence: str) -> None:
    result = check_grounding(sentence, FACTS, PLAN)
    assert result.grounded, result.ungrounded


@pytest.mark.parametrize(
    ("sentence", "bad"),
    [
        ("The Sharpe ratio is 1.83.", "1.83"),  # invented
        ("Volatility is 0.1390.", "0.1390"),  # wrong at the written precision
        ("Volatility is 15%.", "15"),  # wrong percentage
        ("This also happened in 1955.", "1955"),  # year outside the data
        ("On 2030-01-01 it rose.", "2030-01-01"),  # date outside the data
        ("It peaked on 2021-11-18.", "2021-11-18"),  # inside the data, but not a fact
        ("It peaked on 18 November 2021.", "18 November 2021"),  # the same, in words
        ("It peaked on December 3, 2021.", "December 3, 2021"),
        ("It peaked in December 2021.", "December 2021"),  # no fact date in that month
        ("It peaked on 31 February 2021.", "31 February 2021"),  # no such day
        ("This is significant at the 5% level.", "5"),  # no tests, so no significance level
    ],
)
def test_grounding_rejects_invented_numbers(sentence: str, bad: str) -> None:
    result = check_grounding(sentence, FACTS, PLAN)
    assert not result.grounded and result.ungrounded == [bad]


def test_significance_levels_are_grounded_only_when_a_test_ran() -> None:
    sentence = "There is no statistically significant difference at the 5% level (95% interval)."
    assert check_grounding(sentence, FACTS, PLAN).ungrounded == ["5", "95"]
    p_value = Fact(fact_id="s8.p_value", label="p", value=0.4321, display="0.4321")
    assert check_grounding(sentence, [*FACTS, p_value], PLAN).grounded


def test_extra_texts_allow_numbers_from_warnings() -> None:
    sentence = "Aligning the inputs dropped 22% of the rows."
    assert not check_grounding(sentence, FACTS, PLAN).grounded
    warning = "Aligning the inputs kept 2628 of 3380 rows of 'a' (22% dropped)."
    assert check_grounding(sentence, FACTS, PLAN, [warning]).grounded


# --- narrate -----------------------------------------------------------------------


@pytest.mark.parametrize("scenario_id", list(PLANS))
def test_template_narrative_is_fully_grounded(
    scenario_id: str, settings: Settings, catalog: Catalog
) -> None:
    plan, result, issues = run_scenario(scenario_id, catalog)
    narrative = narrate(plan.request, plan, result.artifacts, issues, "mock", settings)
    assert narrative.source == "template"
    assert narrative.grounding.grounded, narrative.grounding.ungrounded
    assert narrative.grounding.numbers_checked > 5


def test_llm_narrative_with_invented_number_is_flagged(
    settings: Settings, catalog: Catalog
) -> None:
    plan, result, issues = run_scenario("spread", catalog)
    facts = build_facts_sheet(plan, result.artifacts)
    latest = next(f.display for f in facts if f.fact_id == "s3.last_value")
    client = FakeAnthropicClient(
        text_response(f"The latest spread is {latest}; the half-life is 7.77.")
    )
    narrative = narrate(plan.request, plan, result.artifacts, issues, "live", settings,
                        client=client)  # fmt: skip
    assert narrative.source == "llm" and narrative.model == "claude-sonnet-5-5"
    assert narrative.grounding.ungrounded == ["7.77"]
    prompt = client.calls[0]["messages"][0]["content"]
    assert "FACTS SHEET" in prompt and latest in prompt
    assert "temperature" not in client.calls[0]


def test_llm_narrative_with_a_wrong_date_fails_grounding(
    settings: Settings, catalog: Catalog
) -> None:
    plan, result, issues = run_scenario("scenario_1", catalog)
    facts = {f.fact_id: f.display for f in build_facts_sheet(plan, result.artifacts)}
    value, latest = facts["s4.last_value"], facts["s4.end"]
    # A week inside the data that is no date of the facts sheet.
    weeks = [f"{d:%Y-%m-%d}" for d in result.artifacts["s4"].value.index]
    wrong = next(week for week in weeks[100:] if week not in facts.values())

    def narrated(text: str):
        client = FakeAnthropicClient(text_response(text))
        return narrate(plan.request, plan, result.artifacts, issues, "live", settings,
                       client=client).grounding  # fmt: skip

    assert narrated(f"Volatility was {value} on {latest}.").grounded
    assert narrated(f"Volatility was {value} on {wrong}.").ungrounded == [wrong]


def test_llm_narrative_with_a_wrong_p_value_fails_grounding(
    settings: Settings, catalog: Catalog
) -> None:
    plan, result, issues = run_scenario("scenario_1", catalog)
    facts = {f.fact_id: f for f in build_facts_sheet(plan, result.artifacts)}
    p_value = facts["s8.p_value"]
    wrong = f"{p_value.value + 0.11:.4f}"  # four decimals, like the displayed value
    episodes = facts["s8.n_episodes_flagged"].display

    def narrated(text: str):
        client = FakeAnthropicClient(text_response(text))
        return narrate(plan.request, plan, result.artifacts, issues, "live", settings,
                       client=client).grounding  # fmt: skip

    right = f"The test gives a p-value of {p_value.display} over {episodes} episodes."
    assert narrated(right).grounded
    bad = narrated(f"The test gives a p-value of {wrong} over {episodes} episodes.")
    assert bad.ungrounded == [wrong]


def hac_artifacts(p_value: float) -> tuple[Plan, dict[str, Artifact]]:
    """A one-step plan and a hand-made hac_mean_difference table with the given p-value."""
    table = {"diff_means": -0.0012, "mean_normal": 0.0021, "se_hac": 0.0019, "t_stat": -0.63,
             "p_value": p_value, "ci_low": -0.0049, "ci_high": 0.0025, "n_obs": 1500,
             "n_flagged": 240, "n_normal": 1260, "n_episodes_flagged": 23, "maxlags_used": 7,
             "horizon": 1, "start": "1996-12-27", "end": "2025-11-14"}  # fmt: skip
    meta = ArtifactMeta(None, Frequency.W, ["NASDAQCOM"], 1500, "1996-12-27", "2025-11-14")
    step = {"id": "s8", "capability": "hac_mean_difference",
            "inputs": {"outcome": "s6", "flags": "s5"}, "params": {}}  # fmt: skip
    plan = Plan(plan_id="p", request="r", steps=[step])
    return plan, {"s8": Artifact(table, DataType.TABLE, meta)}


@pytest.mark.parametrize(
    ("p_value", "verdict"),
    [
        (0.4321, "no statistically significant difference at the 5% level"),
        (0.05, "no statistically significant difference at the 5% level"),  # p >= 0.05
        (0.0123, "a statistically significant difference at the 5% level"),
    ],
)
def test_template_reads_the_test_without_overclaiming(p_value: float, verdict: str) -> None:
    plan, artifacts = hac_artifacts(p_value)
    facts = build_facts_sheet(plan, artifacts)
    text = template_narrative(plan, artifacts, facts, [])
    assert verdict in text
    assert f"p-value of {format_number(p_value)}" in text and "-0.001200" in text
    assert "23 distinct episodes" in text  # always the number of episodes
    assert "no difference" not in text.replace("significant difference", "")
    assert "causes" not in text and "predicts" not in text
    assert check_grounding(text, facts, plan).grounded


def test_llm_failure_falls_back_to_the_template(settings: Settings, catalog: Catalog) -> None:
    plan, result, issues = run_scenario("spread", catalog)
    client = FakeAnthropicClient(error=RuntimeError("connection reset"))
    narrative = narrate(plan.request, plan, result.artifacts, issues, "live", settings,
                        client=client)  # fmt: skip
    assert narrative.source == "template"
    assert "connection reset" in narrative.error
    assert narrative.grounding.grounded


def test_openai_narrative_path(settings: Settings, catalog: Catalog) -> None:
    plan, result, issues = run_scenario("spread", catalog)
    facts = build_facts_sheet(plan, result.artifacts)
    latest = next(f.display for f in facts if f.fact_id == "s3.last_value")
    client = FakeOpenAIClient(openai_text_response(f"The latest spread is {latest}."))
    openai_settings = dataclasses.replace(settings, llm_provider="openai")
    narrative = narrate(plan.request, plan, result.artifacts, issues, "live", openai_settings,
                        client=client)  # fmt: skip
    assert narrative.source == "llm" and narrative.model == "gpt-6-luna"
    assert narrative.grounding.grounded
    call = client.calls[0]
    assert "FACTS SHEET" in call["input"][0]["content"]
    assert call["reasoning"] == {"effort": openai_settings.narrator_effort}
    assert "temperature" not in call


def test_incomplete_openai_narrative_falls_back(settings: Settings, catalog: Catalog) -> None:
    plan, result, issues = run_scenario("spread", catalog)
    client = FakeOpenAIClient(openai_text_response("", "incomplete", "max_output_tokens"))
    openai_settings = dataclasses.replace(settings, llm_provider="openai")
    narrative = narrate(plan.request, plan, result.artifacts, issues, "live", openai_settings,
                        client=client)  # fmt: skip
    assert narrative.source == "template" and "incomplete" in narrative.error
