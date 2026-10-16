"""Static guardrails G01-G13: each triggers on a crafted plan and stays silent on a clean one."""

import copy
import json
from typing import Any

import pytest

from quantlab.data import Catalog
from quantlab.plan import Plan, plan_tool_schema
from quantlab.scenarios import SCENARIOS_BY_ID
from quantlab.validation import (
    GUARDRAILS,
    Severity,
    ValidationReport,
    check_schema,
    fill_defaults,
    planner_failure_issue,
    validate_plan,
)


def scenario_plan(scenario_id: str) -> dict[str, Any]:
    """A deep copy of a canned scenario plan, as a dict we can mutate."""
    scenario = SCENARIOS_BY_ID[scenario_id]
    return {"plan_id": "p-test", "request": scenario.request, **copy.deepcopy(scenario.plan)}


def validate(raw: dict[str, Any], catalog: Catalog, **budgets: int) -> ValidationReport:
    """Parse and validate a plan dict (defaults filled, as the engine does)."""
    return validate_plan(fill_defaults(Plan.model_validate(raw)), catalog, **budgets)


def step(raw: dict[str, Any], step_id: str) -> dict[str, Any]:
    """The step dict with the given id."""
    return next(s for s in raw["steps"] if s["id"] == step_id)


# --- clean cases ----------------------------------------------------------------


@pytest.mark.parametrize("code", [c for c in GUARDRAILS if c.startswith("G")])
def test_clean_plan_triggers_nothing(code: str, catalog: Catalog) -> None:
    # Scenario E is a fully clean plan; A and B only need the return-method approval.
    assert code not in validate(scenario_plan("E"), catalog).codes()
    if code != "G12":
        assert code not in validate(scenario_plan("A"), catalog).codes()
        assert code not in validate(scenario_plan("B"), catalog).codes()


def test_scenario_a_only_needs_the_return_method_approval(catalog: Catalog) -> None:
    report = validate(scenario_plan("A"), catalog)
    assert not report.is_blocked
    assert [i.issue_id for i in report.approvals_required] == ["G12:s2:return_method"]
    assert report.warnings == []


# --- G01 SCHEMA_INVALID ---------------------------------------------------------


def test_g01_extra_fields_are_rejected() -> None:
    raw = scenario_plan("A")
    raw["steps"][0]["code"] = "import os"
    plan, issues = check_schema(raw)
    assert plan is None
    assert {i.code for i in issues} == {"G01"}
    assert all(i.severity == Severity.BLOCK for i in issues)


def test_g01_wrong_structure_and_planner_failure() -> None:
    plan, issues = check_schema({"plan_id": "p", "request": "r", "steps": "not a list"})
    assert plan is None and issues[0].code == "G01"
    assert planner_failure_issue("timeout").code == "G01"
    assert check_schema(scenario_plan("A"))[1] == []


def test_g01_empty_plan(catalog: Catalog) -> None:
    raw = scenario_plan("A")
    raw["steps"] = []
    assert "G01" in validate(raw, catalog).codes()


# --- G02 UNKNOWN_CAPABILITY -----------------------------------------------------


def test_g02_prediction_has_no_capability(catalog: Catalog) -> None:
    report = validate(scenario_plan("C"), catalog)
    assert report.codes() == ["G02"] and report.is_blocked
    assert "forecast" in report.blocks[0].message


def test_g02_unknown_capability(catalog: Catalog) -> None:
    raw = scenario_plan("E")
    step(raw, "s4")["capability"] = "run_python"
    report = validate(raw, catalog)
    assert "G02" in report.codes() and report.is_blocked


# --- G03 INVALID_PARAM ----------------------------------------------------------


@pytest.mark.parametrize(
    ("step_id", "params"),
    [
        ("s3", {"window": 21, "annualize": True, "colour": "red"}),  # unknown param
        ("s3", {"annualize": True}),  # missing required
        ("s3", {"window": "21"}),  # wrong type
        ("s3", {"window": True}),  # bool is not an int
        ("s3", {"window": None}),  # null
        ("s3", {"window": 2}),  # below range
        ("s4", {"quantile": 1.5}),  # above range
        ("s4", {"quantile": float("nan")}),  # NaN
        ("s2", {"method": "arithmetic"}),  # not in choices
        ("s3", {"window": 21, "annualize": "yes"}),  # wrong type for bool
    ],
)
def test_g03_invalid_params(step_id: str, params: dict[str, Any], catalog: Catalog) -> None:
    raw = scenario_plan("B")
    step(raw, step_id)["params"] = params
    report = validate(raw, catalog)
    assert "G03" in report.codes() and report.is_blocked


def test_defaults_are_filled_in() -> None:
    filled = fill_defaults(Plan.model_validate(scenario_plan("B")))
    assert filled.steps[1].params == {"method": "simple", "periods": 1}  # default added
    assert filled.steps[2].params == {"window": 21, "annualize": True}
    unknown = fill_defaults(Plan.model_validate(scenario_plan("C")))
    assert unknown.steps[1].params == {"steps_ahead": 5}  # unknown capability untouched


# --- G04 BAD_REFERENCE ----------------------------------------------------------


@pytest.mark.parametrize(
    "inputs",
    [
        {},  # missing input
        {"series": "s99"},  # unknown step
        {"series": "s4"},  # later step
        {"series": "s3"},  # itself
        {"series": "s2", "other": "s1"},  # unknown input name
    ],
)
def test_g04_bad_references(inputs: dict[str, str], catalog: Catalog) -> None:
    raw = scenario_plan("B")
    step(raw, "s3")["inputs"] = inputs
    assert "G04" in validate(raw, catalog).codes()


def test_g04_duplicate_step_ids(catalog: Catalog) -> None:
    raw = scenario_plan("E")
    step(raw, "s4")["id"] = "s3"
    assert "G04" in validate(raw, catalog).codes()


# --- G05 TYPE_MISMATCH ----------------------------------------------------------


def test_g05_volatility_of_a_price_level(catalog: Catalog) -> None:
    raw = scenario_plan("B")
    step(raw, "s3")["inputs"] = {"series": "s1"}
    report = validate(raw, catalog)
    assert "G05" in report.codes()
    assert "Compute returns" in report.blocks[0].suggestion


def test_g05_summary_statistics_of_flags(catalog: Catalog) -> None:
    raw = scenario_plan("B")
    raw["steps"].append(
        {"id": "s5", "capability": "summary_stats", "inputs": {"series": "s4"}, "params": {}}
    )
    assert "G05" in validate(raw, catalog).codes()


# --- G06 BUDGET_EXCEEDED --------------------------------------------------------


def test_g06_too_many_steps(catalog: Catalog) -> None:
    raw = scenario_plan("E")
    for i in range(5, 17):
        raw["steps"].append(
            {"id": f"s{i}", "capability": "summary_stats", "inputs": {"series": "s3"}}
        )
    assert len(raw["steps"]) == 16
    assert "G06" in validate(raw, catalog).codes()
    assert "G06" not in validate(raw, catalog, max_steps=20).codes()


def test_g06_too_many_datasets_and_long_request(catalog: Catalog) -> None:
    raw = scenario_plan("E")
    assert "G06" in validate(raw, catalog, max_datasets=1).codes()
    raw["request"] = "x" * 3000
    assert "G06" in validate(raw, catalog).codes()


# --- G07 UNKNOWN_DATASET --------------------------------------------------------


def test_g07_unknown_dataset(catalog: Catalog) -> None:
    raw = scenario_plan("E")
    step(raw, "s2")["params"] = {"dataset_id": "BITCOIN"}
    assert "G07" in validate(raw, catalog).codes()
    raw = scenario_plan("E")
    raw["dataset_ids"].append("BITCOIN")
    assert "G07" in validate(raw, catalog).codes()


# --- G08 RETURNS_ON_RATES -------------------------------------------------------


def test_g08_returns_on_rates_replaces_generic_type_error(catalog: Catalog) -> None:
    raw = scenario_plan("E")
    raw["steps"] = [
        step(raw, "s1"),  # the 10-year yield
        {"id": "s2", "capability": "compute_returns", "inputs": {"series": "s1"},
         "params": {"method": "simple"}},
    ]  # fmt: skip
    report = validate(raw, catalog)
    assert "G08" in report.codes() and report.is_blocked
    assert "G05" not in report.codes()  # the specific guardrail wins
    g08 = next(i for i in report.issues if i.code == "G08")
    assert "difference" in g08.suggestion


# --- G09 FREQUENCY_MISMATCH -----------------------------------------------------


def test_g09_spread_of_weekly_and_daily_series(catalog: Catalog) -> None:
    raw = scenario_plan("E")
    raw["steps"].insert(
        2, {"id": "s1w", "capability": "resample", "inputs": {"series": "s1"},
            "params": {"freq": "W"}},
    )  # fmt: skip
    step(raw, "s3")["inputs"] = {"a": "s1w", "b": "s2"}  # weekly 10y minus daily 2y
    report = validate(raw, catalog)
    assert "G09" in report.codes() and report.is_blocked
    assert "resample" in next(i for i in report.issues if i.code == "G09").suggestion


# --- G10 LOOK_AHEAD, G12 MATERIAL_ASSUMPTION -----------------------------------


def test_g10_full_sample_threshold_needs_approval(catalog: Catalog) -> None:
    raw = scenario_plan("B")
    step(raw, "s4")["params"]["method"] = "full_sample"
    report = validate(raw, catalog)
    assert not report.is_blocked
    assert [i.issue_id for i in report.approvals_required] == [
        "G12:s2:return_method",
        "G10:s4",
    ]


def test_g11_message_names_the_chosen_value(catalog: Catalog) -> None:
    raw = scenario_plan("A")
    step(raw, "s2")["params"]["method"] = "log"
    issue = validate(raw, catalog).approvals_required[0]
    assert issue.code == "G12" and "method='log'" in issue.message


# --- G13 PLANNER_REFUSED --------------------------------------------------------


def test_g12_refusal(catalog: Catalog) -> None:
    report = validate(scenario_plan("D"), catalog)
    assert report.codes() == ["G13"] and report.is_blocked


# --- dry run and tool schema ----------------------------------------------------


def test_dry_run_propagates_metadata(catalog: Catalog) -> None:
    dry = validate(scenario_plan("B"), catalog).dry_run
    assert dry["s1"]["kind"] == "price_level" and dry["s1"]["frequency"] == "D"
    assert dry["s2"]["kind"] == "returns" and dry["s2"]["return_method"] == "simple"
    assert dry["s3"]["kind"] == "volatility" and dry["s3"]["dataset_ids"] == ["NASDAQCOM"]
    assert dry["s4"]["data_type"] == "flags"
    assert validate(scenario_plan("A"), catalog).dry_run["s3"]["data_type"] == "table"


def test_tool_schema_excludes_system_fields_and_has_no_refs() -> None:
    schema = plan_tool_schema()
    assert "plan_id" not in schema["properties"] and "request" not in schema["properties"]
    assert "$ref" not in json.dumps(schema)
    step_props = schema["properties"]["steps"]["items"]["properties"]
    assert set(step_props) == {"id", "capability", "inputs", "params", "rationale"}
