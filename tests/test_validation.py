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
    # Scenario F is a fully clean plan; scenario A only needs the return-method approval.
    assert code not in validate(scenario_plan("F"), catalog).codes()
    if code != "G12":
        assert code not in validate(scenario_plan("A"), catalog).codes()


def test_scenario_a_only_needs_the_return_method_approval(catalog: Catalog) -> None:
    report = validate(scenario_plan("A"), catalog)
    assert not report.is_blocked
    assert [i.issue_id for i in report.approvals_required] == ["G12:s3:return_method"]
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


def test_g02_unknown_capability(catalog: Catalog) -> None:
    raw = scenario_plan("F")
    step(raw, "s8")["capability"] = "run_python"
    report = validate(raw, catalog)
    assert "G02" in report.codes() and report.is_blocked


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
    raw = scenario_plan("A")
    step(raw, step_id)["params"] = params
    report = validate(raw, catalog)
    assert "G03" in report.codes() and report.is_blocked


def test_g03_optional_null_is_allowed_and_defaults_are_filled(catalog: Catalog) -> None:
    raw = scenario_plan("A")
    step(raw, "s8")["params"] = {"maxlags": None}
    assert "G03" not in validate(raw, catalog).codes()
    filled = fill_defaults(Plan.model_validate(scenario_plan("C")))
    assert filled.steps[1].params == {"freq": "W", "how": "last"}  # default added
    assert filled.steps[3].params == {"n_estimators": 100}  # unknown capability untouched


# --- G04 BAD_REFERENCE ----------------------------------------------------------


@pytest.mark.parametrize(
    "inputs",
    [
        {},  # missing input
        {"series": "s99"},  # unknown step
        {"series": "s5"},  # later step
        {"series": "s4"},  # itself
        {"series": "s1", "other": "s2"},  # unknown input name
    ],
)
def test_g04_bad_references(inputs: dict[str, str], catalog: Catalog) -> None:
    raw = scenario_plan("A")
    step(raw, "s4")["inputs"] = inputs
    assert "G04" in validate(raw, catalog).codes()


def test_g04_duplicate_step_ids(catalog: Catalog) -> None:
    raw = scenario_plan("F")
    step(raw, "s8")["id"] = "s7"
    assert "G04" in validate(raw, catalog).codes()


# --- G05 TYPE_MISMATCH ----------------------------------------------------------


def test_g05_volatility_of_a_price_level(catalog: Catalog) -> None:
    raw = scenario_plan("A")
    step(raw, "s4")["inputs"] = {"series": "s2"}
    assert "G05" in validate(raw, catalog).codes()


def test_g05_flags_built_on_forward_returns(catalog: Catalog) -> None:
    raw = scenario_plan("A")
    raw["steps"] = [
        step(raw, "s1"),
        step(raw, "s2"),
        step(raw, "s3"),
        {"id": "s4", "capability": "forward_return", "inputs": {"returns": "s3"},
         "params": {"horizon": 4}},
        {"id": "s5", "capability": "flag_regime", "inputs": {"series": "s4"}},
    ]  # fmt: skip
    report = validate(raw, catalog)
    assert "G05" in report.codes()
    assert "never forward returns" in report.blocks[0].suggestion


def test_g05_wrong_data_type(catalog: Catalog) -> None:
    raw = scenario_plan("A")
    raw["steps"].append(
        {"id": "s9", "capability": "summary_stats", "inputs": {"series": "s5"}, "params": {}}
    )
    assert "G05" in validate(raw, catalog).codes()


def test_g05_forward_return_needs_one_period_returns(catalog: Catalog) -> None:
    raw = scenario_plan("A")
    step(raw, "s3")["params"] = {"method": "simple", "periods": 4}
    assert "G05" in validate(raw, catalog).codes()


# --- G06 BUDGET_EXCEEDED --------------------------------------------------------


def test_g06_too_many_steps(catalog: Catalog) -> None:
    raw = scenario_plan("F")
    for i in range(9, 17):
        raw["steps"].append(
            {"id": f"s{i}", "capability": "summary_stats", "inputs": {"series": "s5"}}
        )
    assert len(raw["steps"]) == 16
    assert "G06" in validate(raw, catalog).codes()
    assert "G06" not in validate(raw, catalog, max_steps=20).codes()


def test_g06_too_many_datasets_and_long_request(catalog: Catalog) -> None:
    raw = scenario_plan("F")
    assert "G06" in validate(raw, catalog, max_datasets=1).codes()
    raw["request"] = "x" * 3000
    assert "G06" in validate(raw, catalog).codes()


# --- G07 UNKNOWN_DATASET --------------------------------------------------------


def test_g07_unknown_dataset(catalog: Catalog) -> None:
    raw = scenario_plan("F")
    step(raw, "s2")["params"] = {"dataset_id": "SP500"}
    assert "G07" in validate(raw, catalog).codes()
    raw = scenario_plan("F")
    raw["dataset_ids"].append("BITCOIN")
    assert "G07" in validate(raw, catalog).codes()


# --- G08 RETURNS_ON_RATES -------------------------------------------------------


def test_g08_returns_on_rates_replaces_generic_type_error(catalog: Catalog) -> None:
    report = validate(scenario_plan("C"), catalog)
    assert {"G02", "G08"} <= set(report.codes())
    assert "G05" not in report.codes()  # the specific guardrail wins
    g08 = next(i for i in report.issues if i.code == "G08")
    assert "difference" in g08.suggestion


# --- G09 FREQUENCY_MISMATCH -----------------------------------------------------


def test_g09_frequency_mismatch(catalog: Catalog) -> None:
    report = validate(scenario_plan("E"), catalog)
    assert "G09" in report.codes() and report.is_blocked


# --- G10 LOOK_AHEAD, G11 OVERLAPPING, G12 MATERIAL_ASSUMPTION -------------------


def test_scenario_b_needs_approvals_and_warns(catalog: Catalog) -> None:
    report = validate(scenario_plan("B"), catalog)
    assert not report.is_blocked
    assert {i.code for i in report.approvals_required} == {"G10", "G12"}
    assert [i.code for i in report.warnings] == ["G11"]


def test_g11_is_silenced_by_nonoverlapping_or_one_period_horizon(catalog: Catalog) -> None:
    raw = scenario_plan("B")
    step(raw, "s7")["params"]["nonoverlapping"] = True
    assert "G11" not in validate(raw, catalog).codes()
    raw = scenario_plan("B")
    step(raw, "s6")["params"]["horizon"] = 1
    assert "G11" not in validate(raw, catalog).codes()


def test_g12_message_names_the_chosen_value(catalog: Catalog) -> None:
    raw = scenario_plan("A")
    step(raw, "s3")["params"]["method"] = "log"
    issue = validate(raw, catalog).approvals_required[0]
    assert issue.code == "G12" and "method='log'" in issue.message


# --- G13 PLANNER_REFUSED --------------------------------------------------------


def test_g13_refusal(catalog: Catalog) -> None:
    report = validate(scenario_plan("D"), catalog)
    assert report.codes() == ["G13"] and report.is_blocked


# --- dry run and tool schema ----------------------------------------------------


def test_dry_run_propagates_metadata(catalog: Catalog) -> None:
    dry = validate(scenario_plan("A"), catalog).dry_run
    assert dry["s2"]["frequency"] == "W"
    assert dry["s3"]["kind"] == "returns" and dry["s3"]["return_method"] == "simple"
    assert dry["s5"]["data_type"] == "flags"
    assert dry["s6"]["kind"] == "forward_return" and dry["s6"]["horizon"] == 4
    assert dry["s7"]["data_type"] == "test_result"


def test_tool_schema_excludes_system_fields_and_has_no_refs() -> None:
    schema = plan_tool_schema()
    assert "plan_id" not in schema["properties"] and "request" not in schema["properties"]
    assert "$ref" not in json.dumps(schema)
    step_props = schema["properties"]["steps"]["items"]["properties"]
    assert set(step_props) == {"id", "capability", "inputs", "params", "rationale"}
