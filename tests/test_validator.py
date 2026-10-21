"""The validator's guardrails (V codes, and the P01 refusal block): each triggers on a
crafted plan and stays silent on a clean one. Also the dry run's predicted metadata
and the plan tool schema.
"""

import json
from typing import Any

import pytest

from quantlab.data import Catalog
from quantlab.guardrails import (
    GUARDRAILS,
    MAX_REQUEST_WORDS,
    Action,
    ValidationReport,
)
from quantlab.plan import Plan, plan_tool_schema
from quantlab.validator import check_schema, fill_defaults, planner_failure_issue, validate_plan

from conftest import (
    ARIMA_PLAN,
    SPREAD_PLAN,
    VOLATILITY_APPROVALS,
    look_ahead_plan,
    raw_plan,
    volatility_plan,
)


def validate(raw: dict[str, Any], catalog: Catalog, **budgets: int) -> ValidationReport:
    """Parse and validate a plan dict (defaults filled, as the planner node does)."""
    plan = Plan.model_validate({"plan_id": "p-test", "request": "test request", **raw})
    return validate_plan(fill_defaults(plan), catalog, **budgets)


def step(raw: dict[str, Any], step_id: str) -> dict[str, Any]:
    """The step dict with the given id."""
    return next(s for s in raw["steps"] if s["id"] == step_id)


# --- clean cases ----------------------------------------------------------------


@pytest.mark.parametrize("code", [c for c in GUARDRAILS if c.startswith("V")])
def test_clean_plan_triggers_nothing(code: str, catalog: Catalog) -> None:
    # The spread plan is fully clean; the volatility plan (the anchor of the tests
    # below, see conftest) only needs the return-method approvals.
    assert code not in validate(SPREAD_PLAN, catalog).codes()
    if code != "V10":
        assert code not in validate(volatility_plan(), catalog).codes()


def test_volatility_plan_needs_only_the_return_method_approvals(catalog: Catalog) -> None:
    # One for compute_returns (s3) and one for forward_return (s6); no BLOCK, no WARN.
    report = validate(volatility_plan(), catalog)
    assert not report.is_blocked and report.warnings == []
    assert [i.issue_id for i in report.approvals_required] == VOLATILITY_APPROVALS


# --- V01 INVALID_PLAN: schema, empty plan, step references -----------------------


def test_v01_extra_fields_are_rejected() -> None:
    raw = {"plan_id": "p", "request": "r", **volatility_plan()}
    raw["steps"][0]["code"] = "import os"
    plan, issues = check_schema(raw)
    assert plan is None
    assert {i.code for i in issues} == {"V01"}
    assert all(i.action == Action.BLOCK for i in issues)


def test_v01_wrong_structure_and_planner_failure() -> None:
    plan, issues = check_schema({"plan_id": "p", "request": "r", "steps": "not a list"})
    assert plan is None and issues[0].code == "V01"
    assert planner_failure_issue("timeout").code == "V01"
    assert check_schema({"plan_id": "p", "request": "r", **volatility_plan()})[1] == []


def test_v01_empty_plan(catalog: Catalog) -> None:
    raw = volatility_plan()
    raw["steps"] = []
    assert "V01" in validate(raw, catalog).codes()


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
def test_v01_bad_references(inputs: dict[str, str], catalog: Catalog) -> None:
    raw = volatility_plan()
    step(raw, "s4")["inputs"] = inputs
    report = validate(raw, catalog)
    assert "V01" in report.codes() and report.is_blocked
    assert any(i.issue_id.startswith("V01:s4:") for i in report.blocks)


def test_v01_duplicate_step_ids(catalog: Catalog) -> None:
    raw = volatility_plan()
    step(raw, "s5")["id"] = "s4"
    assert "V01:s4:duplicate" in [i.issue_id for i in validate(raw, catalog).blocks]


# --- V02 UNKNOWN_CAPABILITY -----------------------------------------------------


def test_v02_a_planner_that_invents_a_model_is_still_blocked(catalog: Catalog) -> None:
    report = validate(ARIMA_PLAN, catalog)
    assert report.codes() == ["V02"] and report.is_blocked
    assert "arima" in report.blocks[0].message


# --- V03 INVALID_PARAM ----------------------------------------------------------


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
def test_v03_invalid_params(step_id: str, params: dict[str, Any], catalog: Catalog) -> None:
    raw = volatility_plan()
    step(raw, step_id)["params"] = params
    report = validate(raw, catalog)
    assert "V03" in report.codes() and report.is_blocked


def test_defaults_are_filled_in() -> None:
    raw = volatility_plan()
    del step(raw, "s5")["params"]["min_periods"], step(raw, "s5")["params"]["direction"]
    filled = fill_defaults(Plan(plan_id="p", request="r", **raw))
    assert filled.steps[4].params == {"method": "expanding", "quantile": 0.90,
                                      "min_periods": 52, "direction": "above"}  # added  # fmt: skip
    unknown = fill_defaults(Plan(plan_id="p", request="r", **ARIMA_PLAN))
    assert unknown.steps[1].params == ARIMA_PLAN["steps"][1]["params"]  # left untouched


# --- V04 TYPE_MISMATCH ----------------------------------------------------------


def test_v04_volatility_of_a_price_level(catalog: Catalog) -> None:
    raw = volatility_plan()
    step(raw, "s4")["inputs"] = {"series": "s2"}  # weekly prices, not returns
    report = validate(raw, catalog)
    assert "V04" in report.codes()
    assert "Compute returns" in report.blocks[0].suggestion


def test_v04_flags_are_not_a_series(catalog: Catalog) -> None:
    raw = volatility_plan()
    raw["steps"].append({"id": "s9", "capability": "difference", "inputs": {"series": "s5"},
                         "params": {"periods": 1}})  # fmt: skip
    assert "V04:s9:series" in [i.issue_id for i in validate(raw, catalog).blocks]


def test_v04_compare_groups_needs_flags(catalog: Catalog) -> None:
    raw = volatility_plan()
    raw["steps"].append(
        {"id": "s9", "capability": "compare_groups", "inputs": {"values": "s3", "flags": "s4"}}
    )  # s4 is a volatility series, not flags
    assert "V04" in validate(raw, catalog).codes()


@pytest.mark.parametrize(("outcome", "kind"), [("s2", "price_level"), ("s4", "volatility")])
def test_v04_the_test_needs_a_return_outcome(outcome: str, kind: str, catalog: Catalog) -> None:
    raw = volatility_plan()
    step(raw, "s8")["inputs"]["outcome"] = outcome  # a weekly price level or the volatility
    report = validate(raw, catalog)
    assert [i.issue_id for i in report.blocks] == ["V04:s8:outcome"]
    assert f"kind '{kind}'" in report.blocks[0].message
    assert "non-stationary, which makes the test invalid" in report.blocks[0].message
    assert "forward returns (forward_return)" in report.blocks[0].suggestion
    # Plain returns are a valid outcome too (the same-week comparison).
    step(raw, "s8")["inputs"]["outcome"] = "s3"
    assert "V04" not in validate(raw, catalog).codes()


# --- V05 BUDGET_EXCEEDED --------------------------------------------------------


def test_v05_too_many_steps(catalog: Catalog) -> None:
    raw = volatility_plan()
    for i in range(9, 17):
        raw["steps"].append({"id": f"s{i}", "capability": "difference",
                             "inputs": {"series": "s2"}, "params": {"periods": 1}})  # fmt: skip
    assert len(raw["steps"]) == 16
    assert "V05" in validate(raw, catalog).codes()
    assert "V05" not in validate(raw, catalog, max_steps=20).codes()


def test_the_validator_does_not_check_the_request(catalog: Catalog) -> None:
    # R01 runs before the planner; a request it refuses never gets a plan.
    plan = Plan(plan_id="p", request="word " * (2 * MAX_REQUEST_WORDS), **SPREAD_PLAN)
    assert validate_plan(fill_defaults(plan), catalog).codes() == []


# --- V06 UNKNOWN_DATASET --------------------------------------------------------


def test_v06_unknown_dataset(catalog: Catalog) -> None:
    raw = volatility_plan()
    step(raw, "s1")["params"] = {"dataset_id": "BITCOIN"}
    assert "V06" in validate(raw, catalog).codes()
    raw = volatility_plan()
    raw["dataset_ids"].append("BITCOIN")
    assert "V06" in validate(raw, catalog).codes()


# --- V07 RETURNS_ON_RATES -------------------------------------------------------


def test_v07_returns_on_rates_replaces_generic_type_error(catalog: Catalog) -> None:
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
    assert "V07" in report.codes() and report.is_blocked
    assert "V04" not in report.codes()  # the specific guardrail wins
    v07 = next(i for i in report.issues if i.code == "V07")
    assert "difference" in v07.suggestion


# --- V08 FREQUENCY_MISMATCH -----------------------------------------------------


def test_v08_spread_of_a_weekly_and_a_daily_yield(catalog: Catalog) -> None:
    report = validate(raw_plan("spread_frequency_mismatch_block"), catalog)  # scenario 2
    # The block, and the full-history threshold's look-ahead approval on the flags.
    assert report.is_blocked and report.codes() == ["V08", "V09"]
    (v08,) = report.blocks
    assert v08.step_id == "s4"  # the spread step
    assert "'b' is daily" in v08.message
    assert "Resample DGS2 to weekly" in v08.suggestion


def test_v08_daily_flags_against_weekly_returns(catalog: Catalog) -> None:
    raw = volatility_plan()
    step(raw, "s5")["inputs"] = {"series": "s1"}  # flags on the DAILY prices, not the weekly vol
    report = validate(raw, catalog)
    assert {i.issue_id for i in report.blocks} == {"V08:s7", "V08:s8"}
    assert "'flags' is daily" in report.blocks[0].message


def test_v08_spread_of_weekly_and_daily_series(catalog: Catalog) -> None:
    raw = {"dataset_ids": ["DGS10", "DGS2"], "steps": [dict(s) for s in SPREAD_PLAN["steps"]]}
    raw["steps"].insert(2, {"id": "s1w", "capability": "resample", "inputs": {"series": "s1"},
                            "params": {"freq": "W"}})  # fmt: skip
    step(raw, "s3")["inputs"] = {"a": "s1w", "b": "s2"}  # weekly 10y minus daily 2y
    assert "V08" in validate(raw, catalog).codes()


# --- V09 LOOK_AHEAD, V10 MATERIAL_ASSUMPTION ------------------------------------


def test_v09_full_sample_threshold_needs_approval_and_explains_why(catalog: Catalog) -> None:
    report = validate(look_ahead_plan(), catalog)  # the VIX days of its top 5%, ever
    assert not report.is_blocked
    assert [i.issue_id for i in report.approvals_required] == ["V09:s2"]
    v09 = report.approvals_required[0]
    assert "judges past days using information from the future" in v09.message
    assert "expanding" in v09.suggestion


def test_v09_says_when_it_also_invalidates_the_significance_test(catalog: Catalog) -> None:
    # The volatility plan's flags switched to a full-sample threshold.
    raw = volatility_plan()
    step(raw, "s5")["params"]["method"] = "full_sample"
    report = validate(raw, catalog)
    assert not report.is_blocked
    v09 = next(i for i in report.approvals_required if i.code == "V09")
    assert v09.issue_id == "V09:s5"  # unchanged id: the message is extra, the approval the same
    assert "It also invalidates the significance test in s8" in v09.message
    assert "would not have been" in v09.message
    # Without a test downstream (the look-ahead fixture), it does not mention one.
    plain = validate(look_ahead_plan(), catalog).approvals_required[0]
    assert plain.code == "V09" and "significance test" not in plain.message


def test_v10_message_names_the_chosen_value(catalog: Catalog) -> None:
    issues = validate(volatility_plan(), catalog).approvals_required
    assert [i.code for i in issues] == ["V10", "V10"]
    assert "method='log'" in issues[0].message  # compute_returns, for the volatility
    assert "method='simple'" in issues[1].message  # forward_return applies V10 the same way


# --- P01 PLANNER_REFUSED --------------------------------------------------------


def test_p01_refusal(catalog: Catalog) -> None:
    report = validate(raw_plan("arima_forecast_block"), catalog)
    assert report.codes() == ["P01"] and report.is_blocked


def test_v02_blocks_a_planner_that_follows_an_injection(catalog: Catalog) -> None:
    # Scenario 4 simulates a planner that obeys the injection: a well-formed plan of
    # one step that names a capability that does not exist. Only V02 fires, and it
    # rejects the whole plan.
    report = validate(raw_plan("prompt_injection_unknown_capability"), catalog)
    assert [(i.issue_id, i.action) for i in report.issues] == [("V02:s1", "BLOCK")]
    assert "'run_python' is not a registered capability" in report.issues[0].message
    assert report.is_rejected and report.rejections == report.issues


def test_every_unregistered_step_is_reported_and_the_whole_plan_rejected(
    catalog: Catalog,
) -> None:
    raw = {"dataset_ids": ["NASDAQCOM"], "steps": [
        {"id": "s1", "capability": "load_series", "params": {"dataset_id": "NASDAQCOM"}},
        {"id": "s2", "capability": "run_python", "inputs": {"data": "s1"},
         "params": {"code": "print(1)"}},
        {"id": "s3", "capability": "arima_forecast", "inputs": {"series": "s1"}},
    ]}  # fmt: skip
    report = validate(raw, catalog)
    # Every unregistered step, not just the first; each rejects the plan.
    assert [i.issue_id for i in report.rejections] == ["V02:s2", "V02:s3"]
    assert report.is_rejected


def test_v01_rejects_the_whole_plan(catalog: Catalog) -> None:
    raw = {"dataset_ids": ["NASDAQCOM"], "steps": [
        {"id": "s1", "capability": "resample", "inputs": {"series": "s2"},
         "params": {"freq": "W"}},
        {"id": "s2", "capability": "load_series", "params": {"dataset_id": "NASDAQCOM"}},
    ]}  # fmt: skip
    report = validate(raw, catalog)
    assert {i.code for i in report.rejections} == {"V01"}  # a reference to a later step
    assert report.is_rejected


def test_a_fixable_block_does_not_reject_the_plan(catalog: Catalog) -> None:
    report = validate(raw_plan("spread_frequency_mismatch_block"), catalog)
    assert [i.code for i in report.blocks] == ["V08"]
    assert report.is_blocked and not report.is_rejected


# --- V11 FORWARD_DATA_AS_SIGNAL ---------------------------------------------------


@pytest.mark.parametrize(
    ("capability", "inputs", "params"),
    [
        ("rolling_volatility", {"series": "s6"}, {"window": 12}),
        ("flag_quantile", {"series": "s6"}, {}),
        ("compute_returns", {"series": "s6"}, {"method": "log"}),
        ("difference", {"series": "s6"}, {"periods": 1}),
        ("spread", {"a": "s6", "b": "s3"}, {}),
        ("resample", {"series": "s6"}, {"freq": "M"}),
        ("compare_groups", {"values": "s3", "flags": "s6"}, {}),
        ("hac_mean_difference", {"outcome": "s3", "flags": "s6"}, {}),
    ],
)
def test_v11_forward_returns_are_only_an_outcome(
    capability: str, inputs: dict[str, str], params: dict[str, Any], catalog: Catalog
) -> None:
    raw = volatility_plan()
    raw["steps"].append({"id": "s9", "capability": capability, "inputs": inputs,
                         "params": params})  # fmt: skip
    report = validate(raw, catalog)
    forward_input = next(name for name, ref in inputs.items() if ref == "s6")
    assert [i.issue_id for i in report.blocks] == [f"V11:s9:{forward_input}"]
    assert "can only be used as an outcome, never to build a signal" in report.blocks[0].message
    assert "V04" not in report.codes()  # the specific guardrail replaces the generic one


# --- dry run and tool schema ----------------------------------------------------


def test_dry_run_propagates_metadata(catalog: Catalog) -> None:
    dry = validate(volatility_plan(), catalog).dry_run
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
