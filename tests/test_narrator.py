"""Narrator: facts sheet, formatting, numeric grounding check, template and LLM paths."""

import dataclasses

import pytest

from quantlab.config import Settings
from quantlab.data import Catalog
from quantlab.executor import execute_plan
from quantlab.narrator import (
    Fact,
    build_facts_sheet,
    check_grounding,
    format_number,
    narrate,
)
from quantlab.plan import Plan
from quantlab.scenarios import SCENARIOS_BY_ID
from quantlab.validation import fill_defaults, validate_plan

from fakes import FakeAnthropicClient, FakeOpenAIClient, openai_text_response, text_response


def run_scenario(scenario_id: str, catalog: Catalog):
    """Execute a canned scenario on synthetic data; return plan, result and narrator issues."""
    s = SCENARIOS_BY_ID[scenario_id]
    plan = fill_defaults(Plan(plan_id="p", request=s.request, **s.plan))
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
    plan, result, _ = run_scenario("A", catalog)
    facts = {f.fact_id: f for f in build_facts_sheet(plan, result.artifacts)}
    for fact_id in ["s1.start", "s1.end", "s2.last_value", "s3.mean", "s3.std", "s3.count"]:
        assert fact_id in facts
    assert facts["s3.mean"].value == result.artifacts["s3"].value["mean"]

    plan, result, _ = run_scenario("B", catalog)
    facts = {f.fact_id: f for f in build_facts_sheet(plan, result.artifacts)}
    for fact_id in ["s3.last_value", "s4.n_flagged", "s4.share_flagged", "s4.n_episodes"]:
        assert fact_id in facts
    assert facts["s4.n_flagged"].value == int(result.artifacts["s4"].value.sum())


# --- grounding ---------------------------------------------------------------------

FACTS = [
    Fact(fact_id="s3.last_value", label="vol", value=0.138642, display="0.1386"),
    Fact(fact_id="s3.mean", label="mean", value=0.000563124, display="0.0005631"),
    Fact(fact_id="s4.n_flagged", label="n", value=2832, display="2,832"),
    Fact(fact_id="s1.start", label="start", value="1971-02-05", display="1971-02-05"),
    Fact(fact_id="s1.end", label="end", value="2026-10-13", display="2026-10-13"),
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
        "Data from 1971-02-05 to 2026-10-13, i.e. since 1971.",  # dates and years in range
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
        ("This is significant at the 5% level.", "5"),  # no tests, so no significance level
    ],
)
def test_grounding_rejects_invented_numbers(sentence: str, bad: str) -> None:
    result = check_grounding(sentence, FACTS, PLAN)
    assert not result.grounded and result.ungrounded == [bad]


def test_extra_texts_allow_numbers_from_warnings() -> None:
    sentence = "Aligning the inputs dropped 22% of the rows."
    assert not check_grounding(sentence, FACTS, PLAN).grounded
    warning = "Aligning the inputs kept 2628 of 3380 rows of 'a' (22% dropped)."
    assert check_grounding(sentence, FACTS, PLAN, [warning]).grounded


# --- narrate -----------------------------------------------------------------------


@pytest.mark.parametrize("scenario_id", ["A", "B", "E"])
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
    plan, result, issues = run_scenario("E", catalog)
    facts = build_facts_sheet(plan, result.artifacts)
    mean = next(f.display for f in facts if f.fact_id == "s4.mean")
    client = FakeAnthropicClient(
        text_response(f"The average spread is {mean}; the half-life is 7.77.")
    )
    narrative = narrate(plan.request, plan, result.artifacts, issues, "live", settings,
                        client=client)  # fmt: skip
    assert narrative.source == "llm" and narrative.model == "claude-sonnet-5-5"
    assert narrative.grounding.ungrounded == ["7.77"]
    prompt = client.calls[0]["messages"][0]["content"]
    assert "FACTS SHEET" in prompt and mean in prompt
    assert "temperature" not in client.calls[0]


def test_llm_failure_falls_back_to_the_template(settings: Settings, catalog: Catalog) -> None:
    plan, result, issues = run_scenario("E", catalog)
    client = FakeAnthropicClient(error=RuntimeError("connection reset"))
    narrative = narrate(plan.request, plan, result.artifacts, issues, "live", settings,
                        client=client)  # fmt: skip
    assert narrative.source == "template"
    assert "connection reset" in narrative.error
    assert narrative.grounding.grounded


def test_openai_narrative_path(settings: Settings, catalog: Catalog) -> None:
    plan, result, issues = run_scenario("E", catalog)
    facts = build_facts_sheet(plan, result.artifacts)
    mean = next(f.display for f in facts if f.fact_id == "s4.mean")
    client = FakeOpenAIClient(openai_text_response(f"The average spread is {mean}."))
    openai_settings = dataclasses.replace(settings, llm_provider="openai")
    narrative = narrate(plan.request, plan, result.artifacts, issues, "live", openai_settings,
                        client=client)  # fmt: skip
    assert narrative.source == "llm" and narrative.model == "gpt-6.1-sol"
    assert narrative.grounding.grounded
    call = client.calls[0]
    assert "FACTS SHEET" in call["input"][0]["content"]
    assert call["reasoning"] == {"effort": openai_settings.narrator_effort}
    assert "temperature" not in call


def test_incomplete_openai_narrative_falls_back(settings: Settings, catalog: Catalog) -> None:
    plan, result, issues = run_scenario("E", catalog)
    client = FakeOpenAIClient(openai_text_response("", "incomplete", "max_output_tokens"))
    openai_settings = dataclasses.replace(settings, llm_provider="openai")
    narrative = narrate(plan.request, plan, result.artifacts, issues, "live", openai_settings,
                        client=client)  # fmt: skip
    assert narrative.source == "template" and "incomplete" in narrative.error
