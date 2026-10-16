"""Static guardrails G01-G14: each triggers on a crafted plan and stays silent on a clean one.

Also covers filter_period's interval rules (G03): one way of giving the period,
start before end, valid dates, and a period that overlaps the data.
"""

import json
from typing import Any

import pytest

from quantlab.data import Catalog
from quantlab.guardrails import (
    GUARDRAILS,
    Severity,
    ValidationReport,
    check_schema,
    fill_defaults,
    planner_failure_issue,
    validate_plan,
)
from quantlab.plan import Plan, plan_tool_schema

from conftest import RETURN_METHOD_APPROVAL, SARIMA_PLAN, SPREAD_PLAN, canned_plan, raw_plan

SUCCESS = "volatility_flag_success"  # scenario 1, the anchor case


def validate(raw: dict[str, Any], catalog: Catalog, **budgets: int) -> ValidationReport:
    """Parse and validate a plan dict (defaults filled, as the engine does)."""
    plan = Plan.model_validate({"plan_id": "p-test", "request": "test request", **raw})
    return validate_plan(fill_defaults(plan), catalog, **budgets)


def step(raw: dict[str, Any], step_id: str) -> dict[str, Any]:
    """The step dict with the given id."""
    return next(s for s in raw["steps"] if s["id"] == step_id)


# --- clean cases ----------------------------------------------------------------


@pytest.mark.parametrize("code", [c for c in GUARDRAILS if c.startswith("G")])
def test_clean_plan_triggers_nothing(code: str, catalog: Catalog) -> None:
    # The spread plan is fully clean; scenario 1 only needs the return-method approval.
    assert code not in validate(SPREAD_PLAN, catalog).codes()
    if code != "G12":
        assert code not in validate(raw_plan(SUCCESS), catalog).codes()


def test_scenario_1_needs_exactly_one_approval(catalog: Catalog) -> None:
    report = validate(raw_plan(SUCCESS), catalog)
    assert not report.is_blocked and report.warnings == []
    assert [i.issue_id for i in report.approvals_required] == [RETURN_METHOD_APPROVAL]


# --- G01 SCHEMA_INVALID ---------------------------------------------------------


def test_g01_extra_fields_are_rejected() -> None:
    raw = {"plan_id": "p", "request": "r", **raw_plan(SUCCESS)}
    raw["steps"][0]["code"] = "import os"
    plan, issues = check_schema(raw)
    assert plan is None
    assert {i.code for i in issues} == {"G01"}
    assert all(i.severity == Severity.BLOCK for i in issues)


def test_g01_wrong_structure_and_planner_failure() -> None:
    plan, issues = check_schema({"plan_id": "p", "request": "r", "steps": "not a list"})
    assert plan is None and issues[0].code == "G01"
    assert planner_failure_issue("timeout").code == "G01"
    assert check_schema({"plan_id": "p", "request": "r", **raw_plan(SUCCESS)})[1] == []


def test_g01_empty_plan(catalog: Catalog) -> None:
    raw = raw_plan(SUCCESS)
    raw["steps"] = []
    assert "G01" in validate(raw, catalog).codes()


# --- G02 UNKNOWN_CAPABILITY -----------------------------------------------------


def test_g02_a_planner_that_invents_a_model_is_still_blocked(catalog: Catalog) -> None:
    report = validate(SARIMA_PLAN, catalog)
    assert report.codes() == ["G02"] and report.is_blocked
    assert "sarima" in report.blocks[0].message


# --- G03 INVALID_PARAM ----------------------------------------------------------


@pytest.mark.parametrize(
    ("step_id", "params"),
    [
        ("s4", {"window": 12, "annualize": True, "colour": "red"}),  # unknown param
        ("s4", {"annualize": True}),  # missing required
        ("s4", {"window": "12"}),  # wrong type
        ("s4", {"window": True}),  # bool is not an int
        ("s4", {"window": 2}),  # below range
        ("s5", {"quantile": 1.5}),  # above range
        ("s5", {"quantile": float("nan")}),  # NaN
        ("s3", {"method": "arithmetic"}),  # not in choices
        ("s4", {"window": 12, "annualize": "yes"}),  # wrong type for bool
    ],
)
def test_g03_invalid_params(step_id: str, params: dict[str, Any], catalog: Catalog) -> None:
    raw = raw_plan(SUCCESS)
    step(raw, step_id)["params"] = params
    report = validate(raw, catalog)
    assert "G03" in report.codes() and report.is_blocked


def test_defaults_are_filled_in() -> None:
    raw = raw_plan("frequency_mismatch_block")  # its compute_returns omits `periods`
    filled = fill_defaults(Plan(plan_id="p", request="r", **raw))
    assert filled.steps[4].params == {"method": "simple", "periods": 1}  # default added
    unknown = fill_defaults(Plan(plan_id="p", request="r", **SARIMA_PLAN))
    assert unknown.steps[1].params == SARIMA_PLAN["steps"][1]["params"]  # left untouched


# --- G03 for filter_period: the period must be well defined and inside the data ---
# Synthetic data runs from 1996-01-01 to 2025-11-21 (conftest).


def test_period_plans_are_clean(catalog: Catalog) -> None:
    assert validate(canned_plan("nasdaq_peak_5y"), catalog).issues == []
    since_2020 = validate(canned_plan("nasdaq_volatility_peak_since_2020"), catalog)
    assert since_2020.codes() == ["G12"]


@pytest.mark.parametrize(
    ("params", "issue_id"),
    [
        pytest.param({"lookback_years": 5, "start": "2020-01-01"}, "G03:s2:period", id="both"),
        pytest.param({}, "G03:s2:period", id="neither"),
        pytest.param({"start": "2021-01-01", "end": "2020-01-01"}, "G03:s2:period",
                     id="start after end"),
        pytest.param({"start": "2020/01/01"}, "G03:s2:start", id="not ISO"),
        pytest.param({"start": "2021-02-30"}, "G03:s2:start", id="no such day"),
        pytest.param({"lookback_years": 0}, "G03:s2:lookback_years", id="lookback below range"),
        pytest.param({"lookback_years": 31}, "G03:s2:lookback_years", id="lookback above range"),
        pytest.param({"start": "2031-01-01"}, "G03:s2:coverage", id="after the data"),
        pytest.param({"end": "1990-12-31"}, "G03:s2:coverage", id="before the data"),
    ],
)  # fmt: skip
def test_g03_invalid_periods(params: dict[str, Any], issue_id: str, catalog: Catalog) -> None:
    raw = canned_plan("nasdaq_peak_5y")
    step(raw, "s2")["params"] = params
    report = validate(raw, catalog)
    assert issue_id in [i.issue_id for i in report.blocks]
    assert report.codes() == ["G03"]


def test_a_partly_covered_period_passes(catalog: Catalog) -> None:
    raw = canned_plan("nasdaq_peak_5y")
    step(raw, "s2")["params"] = {"start": "1990-01-01", "end": "1997-06-30"}
    assert validate(raw, catalog).issues == []  # too few rows left would be R02, at run time


def test_null_is_accepted_only_for_nullable_params(catalog: Catalog) -> None:
    raw = canned_plan("nasdaq_peak_5y")
    step(raw, "s2")["params"] = {"lookback_years": 5, "start": None, "end": None}
    assert validate(raw, catalog).issues == []
    raw = raw_plan(SUCCESS)
    step(raw, "s4")["params"] = {"window": None}
    assert "G03:s4:window" in [i.issue_id for i in validate(raw, catalog).blocks]


# --- G04 BAD_REFERENCE ----------------------------------------------------------


@pytest.mark.parametrize(
    "inputs",
    [
        {},  # missing input
        {"series": "s99"},  # unknown step
        {"series": "s5"},  # later step
        {"series": "s4"},  # itself
        {"series": "s3", "other": "s2"},  # unknown input name
    ],
)
def test_g04_bad_references(inputs: dict[str, str], catalog: Catalog) -> None:
    raw = raw_plan(SUCCESS)
    step(raw, "s4")["inputs"] = inputs
    assert "G04" in validate(raw, catalog).codes()


def test_g04_duplicate_step_ids(catalog: Catalog) -> None:
    raw = raw_plan(SUCCESS)
    step(raw, "s5")["id"] = "s4"
    assert "G04" in validate(raw, catalog).codes()


# --- G05 TYPE_MISMATCH ----------------------------------------------------------


def test_g05_volatility_of_a_price_level(catalog: Catalog) -> None:
    raw = raw_plan(SUCCESS)
    step(raw, "s4")["inputs"] = {"series": "s2"}  # weekly prices, not returns
    report = validate(raw, catalog)
    assert "G05" in report.codes()
    assert "Compute returns" in report.blocks[0].suggestion


def test_g05_summary_statistics_of_flags(catalog: Catalog) -> None:
    raw = raw_plan(SUCCESS)
    raw["steps"].append({"id": "s6", "capability": "summary_stats", "inputs": {"series": "s5"}})
    assert "G05" in validate(raw, catalog).codes()


def test_g05_compare_groups_needs_flags(catalog: Catalog) -> None:
    raw = raw_plan(SUCCESS)
    raw["steps"].append(
        {"id": "s6", "capability": "compare_groups", "inputs": {"values": "s3", "flags": "s4"}}
    )  # s4 is a volatility series, not flags
    assert "G05" in validate(raw, catalog).codes()


# --- G06 BUDGET_EXCEEDED --------------------------------------------------------


def test_g06_too_many_steps(catalog: Catalog) -> None:
    raw = raw_plan(SUCCESS)
    for i in range(6, 17):
        raw["steps"].append(
            {"id": f"s{i}", "capability": "summary_stats", "inputs": {"series": "s3"}}
        )
    assert len(raw["steps"]) == 16
    assert "G06" in validate(raw, catalog).codes()
    assert "G06" not in validate(raw, catalog, max_steps=20).codes()


def test_g06_too_many_datasets_and_long_request(catalog: Catalog) -> None:
    assert "G06" in validate(SPREAD_PLAN, catalog, max_datasets=1).codes()
    plan = Plan(plan_id="p", request="x" * 3000, **SPREAD_PLAN)
    assert "G06" in validate_plan(fill_defaults(plan), catalog).codes()


# --- G07 UNKNOWN_DATASET --------------------------------------------------------


def test_g07_unknown_dataset(catalog: Catalog) -> None:
    raw = raw_plan(SUCCESS)
    step(raw, "s1")["params"] = {"dataset_id": "BITCOIN"}
    assert "G07" in validate(raw, catalog).codes()
    raw = raw_plan(SUCCESS)
    raw["dataset_ids"].append("BITCOIN")
    assert "G07" in validate(raw, catalog).codes()


# --- G08 RETURNS_ON_RATES -------------------------------------------------------


def test_g08_returns_on_rates_replaces_generic_type_error(catalog: Catalog) -> None:
    raw = {
        "dataset_ids": ["DGS10"],
        "steps": [
            {"id": "s1", "capability": "load_series", "params": {"dataset_id": "DGS10"}},
            {
                "id": "s2",
                "capability": "compute_returns",
                "inputs": {"series": "s1"},
                "params": {"method": "simple"},
            },
        ],  # fmt: skip
    }
    report = validate(raw, catalog)
    assert "G08" in report.codes() and report.is_blocked
    assert "G05" not in report.codes()  # the specific guardrail wins
    g08 = next(i for i in report.issues if i.code == "G08")
    assert "difference" in g08.suggestion


# --- G09 FREQUENCY_MISMATCH -----------------------------------------------------


def test_g09_daily_vix_flags_against_weekly_returns(catalog: Catalog) -> None:
    report = validate(raw_plan("frequency_mismatch_block"), catalog)
    assert report.is_blocked and report.codes() == ["G09", "G12"]
    g09 = next(i for i in report.issues if i.code == "G09")
    assert g09.step_id == "s6"  # the compare_groups step
    assert "'flags' is daily" in g09.message
    assert "Resample VIXCLS to weekly" in g09.suggestion


def test_g09_spread_of_weekly_and_daily_series(catalog: Catalog) -> None:
    raw = {"dataset_ids": ["DGS10", "DGS2"], "steps": [dict(s) for s in SPREAD_PLAN["steps"]]}
    raw["steps"].insert(2, {"id": "s1w", "capability": "resample", "inputs": {"series": "s1"},
                            "params": {"freq": "W"}})  # fmt: skip
    step(raw, "s3")["inputs"] = {"a": "s1w", "b": "s2"}  # weekly 10y minus daily 2y
    assert "G09" in validate(raw, catalog).codes()


# --- G10 LOOK_AHEAD, G12 MATERIAL_ASSUMPTION ------------------------------------


def test_g10_full_sample_threshold_needs_approval_and_explains_why(catalog: Catalog) -> None:
    report = validate(raw_plan("volatility_flag_lookahead"), catalog)
    assert not report.is_blocked
    assert [i.issue_id for i in report.approvals_required] == [RETURN_METHOD_APPROVAL, "G10:s5"]
    g10 = report.approvals_required[1]
    assert "judges past weeks using information from the future" in g10.message
    assert "expanding" in g10.suggestion


def test_g12_message_names_the_chosen_value(catalog: Catalog) -> None:
    raw = raw_plan(SUCCESS)
    step(raw, "s3")["params"]["method"] = "log"
    issue = validate(raw, catalog).approvals_required[0]
    assert issue.code == "G12" and "method='log'" in issue.message


# --- G13 PLANNER_REFUSED --------------------------------------------------------


@pytest.mark.parametrize("scenario_id", ["sarima_forecast_block", "prompt_injection_block"])
def test_g13_refusal(scenario_id: str, catalog: Catalog) -> None:
    report = validate(raw_plan(scenario_id), catalog)
    assert report.codes() == ["G13"] and report.is_blocked


# --- G14 WINDOW_UNITS_CHANGED ---------------------------------------------------


def edited_plan(changes: dict[str, dict[str, Any]]) -> tuple[Plan, Plan]:
    """(proposal, edited copy) of scenario 1, with `changes` applied to the copy."""
    proposal = fill_defaults(Plan(plan_id="p", request="r", **raw_plan(SUCCESS)))
    edited = proposal.model_copy(deep=True)
    for step in edited.steps:
        step.params.update(changes.get(step.id, {}))
    return proposal, edited


def test_g14_weekly_to_daily_keeps_windows_in_the_wrong_units(catalog: Catalog) -> None:
    proposal, edited = edited_plan({"s2": {"freq": "D"}})
    report = validate_plan(edited, catalog, baseline=proposal)
    g14 = [i for i in report.issues if i.code == "G14"]
    assert [i.issue_id for i in g14] == ["G14:s4:window", "G14:s5:min_periods"]
    assert all(i.severity == Severity.WARN for i in g14) and not report.is_blocked
    assert g14[0].message == (
        "Frequency changed from weekly to daily: window 12 now means 12 trading days, not 12 weeks."
    )
    assert g14[0].suggestion == "Consider 60 for roughly 12 weeks."
    assert report.dry_run["s4"]["frequency"] == "D"  # the dry run follows the edit


def test_g14_is_silent_when_the_window_is_adjusted_or_there_is_no_edit(catalog: Catalog) -> None:
    proposal, edited = edited_plan({"s2": {"freq": "D"}, "s4": {"window": 60}})
    codes = [i.issue_id for i in validate_plan(edited, catalog, baseline=proposal).issues]
    assert "G14:s4:window" not in codes and "G14:s5:min_periods" in codes
    assert "G14" not in validate_plan(edited, catalog).codes()  # no baseline: fresh plan
    assert "G14" not in validate_plan(proposal, catalog, baseline=proposal).codes()


# --- dry run and tool schema ----------------------------------------------------


def test_dry_run_propagates_metadata(catalog: Catalog) -> None:
    dry = validate(raw_plan(SUCCESS), catalog).dry_run
    assert dry["s1"]["frequency"] == "D" and dry["s2"]["frequency"] == "W"
    assert dry["s3"]["kind"] == "returns" and dry["s3"]["return_method"] == "simple"
    assert dry["s4"]["kind"] == "volatility" and dry["s5"]["data_type"] == "flags"
    mismatch = validate(raw_plan("frequency_mismatch_block"), catalog).dry_run
    assert mismatch["s6"]["data_type"] == "table"
    assert mismatch["s6"]["dataset_ids"] == ["NASDAQCOM", "VIXCLS"]
    # filter_period keeps its input's kind and frequency; find_extremes gives a table.
    period = validate(canned_plan("nasdaq_volatility_peak_since_2020"), catalog).dry_run
    assert (period["s5"]["kind"], period["s5"]["frequency"]) == ("volatility", "W")
    assert period["s6"]["data_type"] == "table" and period["s6"]["kind"] is None


def test_tool_schema_excludes_system_fields_and_has_no_refs() -> None:
    schema = plan_tool_schema()
    assert "plan_id" not in schema["properties"] and "request" not in schema["properties"]
    assert "$ref" not in json.dumps(schema)
    step_props = schema["properties"]["steps"]["items"]["properties"]
    assert set(step_props) == {"id", "capability", "inputs", "params", "rationale"}
