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
    ("key", "value", "expected"),
    [
        ("p_value", 0.123456, "0.1235"),
        ("p_value", 0.00001, "<0.0001"),
        ("mean", 0.001234567, "0.001235"),
        ("last_value", 27204.38, "27,204"),
        ("std", 0.9145813, "0.9146"),
        ("min", -2.34, "-2.340"),
        ("n_obs", 2832, "2,832"),
        ("mean", 0.0, "0"),
    ],
)
def test_format_number(key: str, value: float, expected: str) -> None:
    assert format_number(key, value) == expected


def test_facts_sheet_covers_tests_and_series(catalog: Catalog) -> None:
    plan, result, _ = run_scenario("A", catalog)
    facts = {f.fact_id: f for f in build_facts_sheet(plan, result.artifacts)}
    for fact_id in ["s4.last_value", "s5.n_episodes", "s7.p_value", "s7.diff_means",
                    "s8.coef_flag", "s8.maxlags"]:  # fmt: skip
        assert fact_id in facts
    assert facts["s7.p_value"].value == result.artifacts["s7"].value["p_value"]
    assert "nonoverlapping" not in " ".join(facts)  # booleans are not facts


# --- grounding ---------------------------------------------------------------------

FACTS = [
    Fact(fact_id="s4.last_value", label="vol", value=0.138642, display="0.1386"),
    Fact(fact_id="s7.p_value", label="p", value=0.50093, display="0.5009"),
    Fact(fact_id="s7.n_flagged", label="n", value=2832, display="2,832"),
    Fact(fact_id="s7.tiny_p", label="p", value=3e-9, display="<0.0001"),
    Fact(fact_id="s1.start", label="start", value="1971-02-05", display="1971-02-05"),
    Fact(fact_id="s1.end", label="end", value="2026-10-13", display="2026-10-13"),
]
PLAN = Plan(
    plan_id="p",
    request="r",
    steps=[{"id": "s5", "capability": "flag_regime", "params": {"quantile": 0.9, "window": 12}}],
)


@pytest.mark.parametrize(
    "sentence",
    [
        "Volatility is 0.1386 and the p-value is 0.5009.",  # exactly as displayed
        "Volatility is 13.86% today.",  # percent form
        "Volatility is about 14% and 0.14.",  # coarser but correct rounding
        "There were 2,832 flagged weeks, and 2832 is a lot.",  # thousands separator
        "The p-value is below 0.0001.",  # displayed threshold
        "Weeks above the 90th percentile with a 12-week window.",  # plan parameters
        "Data from 1971-02-05 to 2026-10-13, i.e. since 1971.",  # dates and years in range
        "Not significant at the 5% level.",  # conventional level
        "Steps s7 and R04 and DGS10 are named, not numbers.",  # ids are skipped
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


@pytest.mark.parametrize("scenario_id", ["A", "B", "F"])
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
    plan, result, issues = run_scenario("F", catalog)
    facts = build_facts_sheet(plan, result.artifacts)
    p_value = next(f.display for f in facts if f.fact_id == "s8.p_value")
    client = FakeAnthropicClient(
        text_response(f"The ADF p-value is {p_value}; the half-life is 7.77.")
    )
    narrative = narrate(plan.request, plan, result.artifacts, issues, "live", settings,
                        client=client)  # fmt: skip
    assert narrative.source == "llm" and narrative.model == "claude-sonnet-5-5"
    assert narrative.grounding.ungrounded == ["7.77"]
    prompt = client.calls[0]["messages"][0]["content"]
    assert "FACTS SHEET" in prompt and p_value in prompt
    assert "temperature" not in client.calls[0]


def test_llm_failure_falls_back_to_the_template(settings: Settings, catalog: Catalog) -> None:
    plan, result, issues = run_scenario("F", catalog)
    client = FakeAnthropicClient(error=RuntimeError("connection reset"))
    narrative = narrate(plan.request, plan, result.artifacts, issues, "live", settings,
                        client=client)  # fmt: skip
    assert narrative.source == "template"
    assert "connection reset" in narrative.error
    assert narrative.grounding.grounded


def test_openai_narrative_path(settings: Settings, catalog: Catalog) -> None:
    plan, result, issues = run_scenario("F", catalog)
    facts = build_facts_sheet(plan, result.artifacts)
    p_value = next(f.display for f in facts if f.fact_id == "s8.p_value")
    client = FakeOpenAIClient(openai_text_response(f"The ADF p-value is {p_value}."))
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
    plan, result, issues = run_scenario("F", catalog)
    client = FakeOpenAIClient(openai_text_response("", "incomplete", "max_output_tokens"))
    openai_settings = dataclasses.replace(settings, llm_provider="openai")
    narrative = narrate(plan.request, plan, result.artifacts, issues, "live", openai_settings,
                        client=client)  # fmt: skip
    assert narrative.source == "template" and "incomplete" in narrative.error
