"""Static guardrails (G codes): each triggers on a crafted plan and stays silent on a clean one.

Also checks that the GUARDRAILS table (shown on the Guardrails page) lists
exactly the implemented checks.
"""

import ast
import json
from typing import Any

import pytest

from quantlab.config import PROJECT_ROOT
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

from conftest import (
    ARIMA_PLAN,
    SCENARIO_1_APPROVALS,
    SPREAD_PLAN,
    raw_plan,
)

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
    # The spread plan is fully clean; scenario 1 only needs the return-method approvals.
    assert code not in validate(SPREAD_PLAN, catalog).codes()
    if code != "G10":
        assert code not in validate(raw_plan(SUCCESS), catalog).codes()


def test_scenario_1_needs_only_the_return_method_approvals(catalog: Catalog) -> None:
    # One for compute_returns (s3) and one for forward_return (s6); no BLOCK, no WARN.
    report = validate(raw_plan(SUCCESS), catalog)
    assert not report.is_blocked and report.warnings == []
    assert [i.issue_id for i in report.approvals_required] == SCENARIO_1_APPROVALS


# --- G01 INVALID_PLAN: schema, empty plan, step references -----------------------


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
def test_g01_bad_references(inputs: dict[str, str], catalog: Catalog) -> None:
    raw = raw_plan(SUCCESS)
    step(raw, "s4")["inputs"] = inputs
    report = validate(raw, catalog)
    assert "G01" in report.codes() and report.is_blocked
    assert any(i.issue_id.startswith("G01:s4:") for i in report.blocks)


def test_g01_duplicate_step_ids(catalog: Catalog) -> None:
    raw = raw_plan(SUCCESS)
    step(raw, "s5")["id"] = "s4"
    assert "G01:s4:duplicate" in [i.issue_id for i in validate(raw, catalog).blocks]


# --- G02 UNKNOWN_CAPABILITY -----------------------------------------------------


def test_g02_a_planner_that_invents_a_model_is_still_blocked(catalog: Catalog) -> None:
    report = validate(ARIMA_PLAN, catalog)
    assert report.codes() == ["G02"] and report.is_blocked
    assert "arima" in report.blocks[0].message


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
        ("s4", {"window": None}),  # null is never a valid value
        ("s6", {"method": "log", "horizon": 4}),  # forward returns are always one period
        ("s8", {"maxlags": 4}),  # the Newey-West lag is always set by its rule
    ],
)
def test_g03_invalid_params(step_id: str, params: dict[str, Any], catalog: Catalog) -> None:
    raw = raw_plan(SUCCESS)
    step(raw, step_id)["params"] = params
    report = validate(raw, catalog)
    assert "G03" in report.codes() and report.is_blocked


def test_defaults_are_filled_in() -> None:
    raw = raw_plan("vix_full_sample_lookahead")  # its flag_regime omits `min_periods`
    filled = fill_defaults(Plan(plan_id="p", request="r", **raw))
    assert filled.steps[1].params == {"quantile": 0.95, "method": "full_sample",
                                      "min_periods": 52}  # default added  # fmt: skip
    unknown = fill_defaults(Plan(plan_id="p", request="r", **ARIMA_PLAN))
    assert unknown.steps[1].params == ARIMA_PLAN["steps"][1]["params"]  # left untouched


# --- G04 TYPE_MISMATCH ----------------------------------------------------------


def test_g04_volatility_of_a_price_level(catalog: Catalog) -> None:
    raw = raw_plan(SUCCESS)
    step(raw, "s4")["inputs"] = {"series": "s2"}  # weekly prices, not returns
    report = validate(raw, catalog)
    assert "G04" in report.codes()
    assert "Compute returns" in report.blocks[0].suggestion


def test_g04_flags_are_not_a_series(catalog: Catalog) -> None:
    raw = raw_plan(SUCCESS)
    raw["steps"].append({"id": "s9", "capability": "difference", "inputs": {"series": "s5"},
                         "params": {"periods": 1}})  # fmt: skip
    assert "G04:s9:series" in [i.issue_id for i in validate(raw, catalog).blocks]


def test_g04_compare_groups_needs_flags(catalog: Catalog) -> None:
    raw = raw_plan(SUCCESS)
    raw["steps"].append(
        {"id": "s9", "capability": "compare_groups", "inputs": {"values": "s3", "flags": "s4"}}
    )  # s4 is a volatility series, not flags
    assert "G04" in validate(raw, catalog).codes()


@pytest.mark.parametrize(("outcome", "kind"), [("s2", "price_level"), ("s4", "volatility")])
def test_g04_the_test_needs_a_return_outcome(outcome: str, kind: str, catalog: Catalog) -> None:
    raw = raw_plan(SUCCESS)
    step(raw, "s8")["inputs"]["outcome"] = outcome  # a weekly price level or the volatility
    report = validate(raw, catalog)
    assert [i.issue_id for i in report.blocks] == ["G04:s8:outcome"]
    assert f"kind '{kind}'" in report.blocks[0].message
    assert "non-stationary, which makes the test invalid" in report.blocks[0].message
    assert "forward returns (forward_return)" in report.blocks[0].suggestion
    # Plain returns are a valid outcome too (the same-week comparison).
    step(raw, "s8")["inputs"]["outcome"] = "s3"
    assert "G04" not in validate(raw, catalog).codes()


# --- G05 BUDGET_EXCEEDED --------------------------------------------------------


def test_g05_too_many_steps(catalog: Catalog) -> None:
    raw = raw_plan(SUCCESS)
    for i in range(9, 17):
        raw["steps"].append({"id": f"s{i}", "capability": "difference",
                             "inputs": {"series": "s2"}, "params": {"periods": 1}})  # fmt: skip
    assert len(raw["steps"]) == 16
    assert "G05" in validate(raw, catalog).codes()
    assert "G05" not in validate(raw, catalog, max_steps=20).codes()


def test_g05_too_many_datasets_and_long_request(catalog: Catalog) -> None:
    assert "G05" in validate(SPREAD_PLAN, catalog, max_datasets=1).codes()
    plan = Plan(plan_id="p", request="x" * 3000, **SPREAD_PLAN)
    assert "G05" in validate_plan(fill_defaults(plan), catalog).codes()


# --- G06 UNKNOWN_DATASET --------------------------------------------------------


def test_g06_unknown_dataset(catalog: Catalog) -> None:
    raw = raw_plan(SUCCESS)
    step(raw, "s1")["params"] = {"dataset_id": "BITCOIN"}
    assert "G06" in validate(raw, catalog).codes()
    raw = raw_plan(SUCCESS)
    raw["dataset_ids"].append("BITCOIN")
    assert "G06" in validate(raw, catalog).codes()


# --- G07 RETURNS_ON_RATES -------------------------------------------------------


def test_g07_returns_on_rates_replaces_generic_type_error(catalog: Catalog) -> None:
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
    assert "G07" in report.codes() and report.is_blocked
    assert "G04" not in report.codes()  # the specific guardrail wins
    g07 = next(i for i in report.issues if i.code == "G07")
    assert "difference" in g07.suggestion


# --- G08 FREQUENCY_MISMATCH -----------------------------------------------------


def test_g08_spread_of_a_weekly_and_a_daily_yield(catalog: Catalog) -> None:
    report = validate(raw_plan("spread_frequency_mismatch_block"), catalog)  # scenario 5
    assert report.is_blocked and report.codes() == ["G08"]
    (g08,) = report.issues
    assert g08.step_id == "s4"  # the spread step
    assert "'b' is daily" in g08.message
    assert "Resample DGS2 to weekly" in g08.suggestion


def test_g08_daily_flags_against_weekly_returns(catalog: Catalog) -> None:
    raw = raw_plan(SUCCESS)
    step(raw, "s5")["inputs"] = {"series": "s1"}  # flags on the DAILY prices, not the weekly vol
    report = validate(raw, catalog)
    assert {i.issue_id for i in report.blocks} == {"G08:s7", "G08:s8"}
    assert "'flags' is daily" in report.blocks[0].message


def test_g08_spread_of_weekly_and_daily_series(catalog: Catalog) -> None:
    raw = {"dataset_ids": ["DGS10", "DGS2"], "steps": [dict(s) for s in SPREAD_PLAN["steps"]]}
    raw["steps"].insert(2, {"id": "s1w", "capability": "resample", "inputs": {"series": "s1"},
                            "params": {"freq": "W"}})  # fmt: skip
    step(raw, "s3")["inputs"] = {"a": "s1w", "b": "s2"}  # weekly 10y minus daily 2y
    assert "G08" in validate(raw, catalog).codes()


# --- G09 LOOK_AHEAD, G10 MATERIAL_ASSUMPTION ------------------------------------


def test_g09_full_sample_threshold_needs_approval_and_explains_why(catalog: Catalog) -> None:
    report = validate(raw_plan("vix_full_sample_lookahead"), catalog)  # scenario 2
    assert not report.is_blocked
    assert [i.issue_id for i in report.approvals_required] == ["G09:s2"]
    g09 = report.approvals_required[0]
    assert "judges past days using information from the future" in g09.message
    assert "expanding" in g09.suggestion


def test_g09_says_when_it_also_invalidates_the_significance_test(catalog: Catalog) -> None:
    # The flagship demo edit: scenario 1's flags switched to a full-sample threshold.
    raw = raw_plan(SUCCESS)
    step(raw, "s5")["params"]["method"] = "full_sample"
    report = validate(raw, catalog)
    assert not report.is_blocked
    g09 = next(i for i in report.approvals_required if i.code == "G09")
    assert g09.issue_id == "G09:s5"  # unchanged id: the message is extra, the approval the same
    assert "It also invalidates the significance test in s8" in g09.message
    assert "would not have been" in g09.message
    # Without a test downstream (scenario 2), the message does not mention one.
    plain = validate(raw_plan("vix_full_sample_lookahead"), catalog).approvals_required[0]
    assert "significance test" not in plain.message


def test_g10_message_names_the_chosen_value(catalog: Catalog) -> None:
    issues = validate(raw_plan(SUCCESS), catalog).approvals_required
    assert [i.code for i in issues] == ["G10", "G10"]
    assert "method='log'" in issues[0].message  # compute_returns, for the volatility
    assert "method='simple'" in issues[1].message  # forward_return applies G10 the same way


# --- G11 PLANNER_REFUSED --------------------------------------------------------


@pytest.mark.parametrize("scenario_id", ["arima_forecast_block", "prompt_injection_block"])
def test_g11_refusal(scenario_id: str, catalog: Catalog) -> None:
    report = validate(raw_plan(scenario_id), catalog)
    assert report.codes() == ["G11"] and report.is_blocked


# --- G12 FORWARD_DATA_AS_SIGNAL ---------------------------------------------------


@pytest.mark.parametrize(
    ("capability", "inputs", "params"),
    [
        ("rolling_volatility", {"series": "s6"}, {"window": 12}),
        ("flag_regime", {"series": "s6"}, {}),
        ("compute_returns", {"series": "s6"}, {"method": "log"}),
        ("difference", {"series": "s6"}, {"periods": 1}),
        ("spread", {"a": "s6", "b": "s3"}, {}),
        ("resample", {"series": "s6"}, {"freq": "M"}),
        ("compare_groups", {"values": "s3", "flags": "s6"}, {}),
        ("hac_mean_difference", {"outcome": "s3", "flags": "s6"}, {}),
    ],
)
def test_g12_forward_returns_are_only_an_outcome(
    capability: str, inputs: dict[str, str], params: dict[str, Any], catalog: Catalog
) -> None:
    raw = raw_plan(SUCCESS)
    raw["steps"].append({"id": "s9", "capability": capability, "inputs": inputs,
                         "params": params})  # fmt: skip
    report = validate(raw, catalog)
    forward_input = next(name for name, ref in inputs.items() if ref == "s6")
    assert [i.issue_id for i in report.blocks] == [f"G12:s9:{forward_input}"]
    assert "can only be used as an outcome, never to build a signal" in report.blocks[0].message
    assert "G04" not in report.codes()  # the specific guardrail replaces the generic one


# --- dry run and tool schema ----------------------------------------------------


def test_dry_run_propagates_metadata(catalog: Catalog) -> None:
    dry = validate(raw_plan(SUCCESS), catalog).dry_run
    assert dry["s1"]["frequency"] == "D" and dry["s2"]["frequency"] == "W"
    assert dry["s3"]["kind"] == "returns" and dry["s3"]["return_method"] == "log"
    assert dry["s4"]["kind"] == "volatility" and dry["s5"]["data_type"] == "flags"
    assert (dry["s6"]["kind"], dry["s6"]["horizon"]) == ("forward_returns", 1)
    assert (dry["s8"]["data_type"], dry["s8"]["horizon"]) == ("table", 1)  # from the outcome
    mismatch = validate(raw_plan("spread_frequency_mismatch_block"), catalog).dry_run
    assert (mismatch["s2"]["frequency"], mismatch["s3"]["frequency"]) == ("W", "D")
    assert (mismatch["s4"]["kind"], mismatch["s4"]["dataset_ids"]) == ("spread", ["DGS10", "DGS2"])


def test_tool_schema_excludes_system_fields_and_has_no_refs() -> None:
    schema = plan_tool_schema()
    assert "plan_id" not in schema["properties"] and "request" not in schema["properties"]
    assert "$ref" not in json.dumps(schema)
    step_props = schema["properties"]["steps"]["items"]["properties"]
    assert set(step_props) == {"id", "capability", "inputs", "params", "rationale"}


# --- the GUARDRAILS table lists exactly the implemented checks --------------------
# The Guardrails page and its counts are generated from the table, so it must not
# drift from the code: every issue is built by make_issue(code, ...), and the codes
# are read from the source of every module.


def is_call_to(node: ast.AST, name: str) -> bool:
    """Whether a syntax-tree node is a call to the plain name `name`, e.g. make_issue(...)."""
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name


def make_issue_codes() -> set[tuple[str, str]]:
    """(file name, guardrail code) for every make_issue call in src/quantlab."""
    found = set()
    for path in sorted((PROJECT_ROOT / "src" / "quantlab").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if is_call_to(node, "make_issue"):
                code = node.args[0]
                assert isinstance(code, ast.Constant), f"{path.name}:{node.lineno}: literal code"
                found.add((path.name, code.value))
    return found


def test_guardrail_table_matches_the_implemented_checks() -> None:
    raised = make_issue_codes()
    # Static checks live in guardrails.py, runtime checks in executor.py, nowhere else.
    assert {file for file, _ in raised} == {"guardrails.py", "executor.py"}
    static = {code for file, code in raised if file == "guardrails.py"}
    runtime = {code for file, code in raised if file == "executor.py"}
    assert static == {c for c, g in GUARDRAILS.items() if g.stage == "static"}
    assert runtime == {c for c, g in GUARDRAILS.items() if g.stage == "runtime"}
    # Static guardrails are the plan-level G codes, runtime ones the R codes.
    assert all(code.startswith("G") for code in static)
    assert all(code.startswith("R") for code in runtime)


def test_issues_are_only_built_through_make_issue() -> None:
    """So the make_issue scan above sees every guardrail code."""
    builders = []
    for path in sorted((PROJECT_ROOT / "src" / "quantlab").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef) and any(
                is_call_to(inner, "Issue") for inner in ast.walk(node)
            ):
                builders.append((path.name, node.name))
    assert builders == [("guardrails.py", "make_issue")]
