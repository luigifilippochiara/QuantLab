"""The request check (R01) and the validator's guardrails (V codes): each triggers on a
crafted request or plan and stays silent on a clean one.

Also checks that the GUARDRAILS table (shown on the Guardrails page) lists
exactly the implemented checks, that each code's prefix names who enforces it,
that codes are never renumbered, and that the descriptions quote the executor's
thresholds.
"""

import ast
import dataclasses
import json
import re
from typing import Any

import pytest

from quantlab.config import PROJECT_ROOT, Settings
from quantlab.data import Catalog
from quantlab.executor import (
    ALIGNMENT_LOSS_THRESHOLD,
    MIN_FLAGGED_EPISODES,
    MISSING_DATA_THRESHOLD,
)
from quantlab.guardrails import (
    CATEGORY_DEFINITIONS,
    ENFORCERS,
    GUARDRAILS,
    LLM_STAGES,
    MAX_REQUEST_CHARS,
    MAX_REQUEST_WORDS,
    Action,
    Category,
    GuardrailType,
    Issue,
    ValidationReport,
    check_request_length,
    check_schema,
    fill_defaults,
    planner_failure_issue,
    validate_plan,
)
from quantlab.mock_scenarios import FLAGSHIP_REQUEST
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


@pytest.mark.parametrize("code", [c for c in GUARDRAILS if c.startswith("V")])
def test_clean_plan_triggers_nothing(code: str, catalog: Catalog) -> None:
    # The spread plan is fully clean; scenario 1 only needs the return-method approvals.
    assert code not in validate(SPREAD_PLAN, catalog).codes()
    if code != "V10":
        assert code not in validate(raw_plan(SUCCESS), catalog).codes()


def test_scenario_1_needs_only_the_return_method_approvals(catalog: Catalog) -> None:
    # One for compute_returns (s3) and one for forward_return (s6); no BLOCK, no WARN.
    report = validate(raw_plan(SUCCESS), catalog)
    assert not report.is_blocked and report.warnings == []
    assert [i.issue_id for i in report.approvals_required] == SCENARIO_1_APPROVALS


# --- R01 REQUEST_TOO_LONG (the request check, before the planner) ----------------


def test_r01_passes_a_normal_request_and_one_at_the_limits() -> None:
    assert check_request_length(FLAGSHIP_REQUEST) == []
    assert check_request_length(" ".join(["word"] * MAX_REQUEST_WORDS)) == []
    assert check_request_length("x" * MAX_REQUEST_CHARS) == []


@pytest.mark.parametrize(
    ("text", "counted"),
    [
        (" ".join(["word"] * (MAX_REQUEST_WORDS + 1)), "1,001 words and 5,004 characters"),
        ("x" * (MAX_REQUEST_CHARS + 1), "1 words and 8,001 characters"),  # no spaces to count
    ],
)
def test_r01_blocks_too_many_words_or_characters(text: str, counted: str) -> None:
    [issue] = check_request_length(text)
    assert (issue.code, issue.issue_id, issue.action) == ("R01", "R01:plan:request", "BLOCK")
    assert (issue.enforced_by, issue.type) == ("Request check", "Deterministic")
    assert f"The request has {counted}; the limits are 1,000 words and 8,000" in issue.message


# --- V01 INVALID_PLAN: schema, empty plan, step references -----------------------


def test_v01_extra_fields_are_rejected() -> None:
    raw = {"plan_id": "p", "request": "r", **raw_plan(SUCCESS)}
    raw["steps"][0]["code"] = "import os"
    plan, issues = check_schema(raw)
    assert plan is None
    assert {i.code for i in issues} == {"V01"}
    assert all(i.action == Action.BLOCK for i in issues)


def test_v01_wrong_structure_and_planner_failure() -> None:
    plan, issues = check_schema({"plan_id": "p", "request": "r", "steps": "not a list"})
    assert plan is None and issues[0].code == "V01"
    assert planner_failure_issue("timeout").code == "V01"
    assert check_schema({"plan_id": "p", "request": "r", **raw_plan(SUCCESS)})[1] == []


def test_v01_empty_plan(catalog: Catalog) -> None:
    raw = raw_plan(SUCCESS)
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
    raw = raw_plan(SUCCESS)
    step(raw, "s4")["inputs"] = inputs
    report = validate(raw, catalog)
    assert "V01" in report.codes() and report.is_blocked
    assert any(i.issue_id.startswith("V01:s4:") for i in report.blocks)


def test_v01_duplicate_step_ids(catalog: Catalog) -> None:
    raw = raw_plan(SUCCESS)
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
    raw = raw_plan(SUCCESS)
    step(raw, step_id)["params"] = params
    report = validate(raw, catalog)
    assert "V03" in report.codes() and report.is_blocked


def test_defaults_are_filled_in() -> None:
    raw = raw_plan("vix_full_sample_lookahead")  # its flag_regime omits `min_periods`
    filled = fill_defaults(Plan(plan_id="p", request="r", **raw))
    assert filled.steps[1].params == {"quantile": 0.95, "method": "full_sample",
                                      "min_periods": 52}  # default added  # fmt: skip
    unknown = fill_defaults(Plan(plan_id="p", request="r", **ARIMA_PLAN))
    assert unknown.steps[1].params == ARIMA_PLAN["steps"][1]["params"]  # left untouched


# --- V04 TYPE_MISMATCH ----------------------------------------------------------


def test_v04_volatility_of_a_price_level(catalog: Catalog) -> None:
    raw = raw_plan(SUCCESS)
    step(raw, "s4")["inputs"] = {"series": "s2"}  # weekly prices, not returns
    report = validate(raw, catalog)
    assert "V04" in report.codes()
    assert "Compute returns" in report.blocks[0].suggestion


def test_v04_flags_are_not_a_series(catalog: Catalog) -> None:
    raw = raw_plan(SUCCESS)
    raw["steps"].append({"id": "s9", "capability": "difference", "inputs": {"series": "s5"},
                         "params": {"periods": 1}})  # fmt: skip
    assert "V04:s9:series" in [i.issue_id for i in validate(raw, catalog).blocks]


def test_v04_compare_groups_needs_flags(catalog: Catalog) -> None:
    raw = raw_plan(SUCCESS)
    raw["steps"].append(
        {"id": "s9", "capability": "compare_groups", "inputs": {"values": "s3", "flags": "s4"}}
    )  # s4 is a volatility series, not flags
    assert "V04" in validate(raw, catalog).codes()


@pytest.mark.parametrize(("outcome", "kind"), [("s2", "price_level"), ("s4", "volatility")])
def test_v04_the_test_needs_a_return_outcome(outcome: str, kind: str, catalog: Catalog) -> None:
    raw = raw_plan(SUCCESS)
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
    raw = raw_plan(SUCCESS)
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
    raw = raw_plan(SUCCESS)
    step(raw, "s1")["params"] = {"dataset_id": "BITCOIN"}
    assert "V06" in validate(raw, catalog).codes()
    raw = raw_plan(SUCCESS)
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
    assert report.is_blocked and report.codes() == ["V08"]
    (v08,) = report.issues
    assert v08.step_id == "s4"  # the spread step
    assert "'b' is daily" in v08.message
    assert "Resample DGS2 to weekly" in v08.suggestion


def test_v08_daily_flags_against_weekly_returns(catalog: Catalog) -> None:
    raw = raw_plan(SUCCESS)
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
    report = validate(raw_plan("vix_full_sample_lookahead"), catalog)  # scenario 5
    assert not report.is_blocked
    assert [i.issue_id for i in report.approvals_required] == ["V09:s2"]
    v09 = report.approvals_required[0]
    assert "judges past days using information from the future" in v09.message
    assert "expanding" in v09.suggestion


def test_v09_says_when_it_also_invalidates_the_significance_test(catalog: Catalog) -> None:
    # The flagship demo edit: scenario 1's flags switched to a full-sample threshold.
    raw = raw_plan(SUCCESS)
    step(raw, "s5")["params"]["method"] = "full_sample"
    report = validate(raw, catalog)
    assert not report.is_blocked
    v09 = next(i for i in report.approvals_required if i.code == "V09")
    assert v09.issue_id == "V09:s5"  # unchanged id: the message is extra, the approval the same
    assert "It also invalidates the significance test in s8" in v09.message
    assert "would not have been" in v09.message
    # Without a test downstream (scenario 5), the message does not mention one.
    plain = validate(raw_plan("vix_full_sample_lookahead"), catalog).approvals_required[0]
    assert "significance test" not in plain.message


def test_v10_message_names_the_chosen_value(catalog: Catalog) -> None:
    issues = validate(raw_plan(SUCCESS), catalog).approvals_required
    assert [i.code for i in issues] == ["V10", "V10"]
    assert "method='log'" in issues[0].message  # compute_returns, for the volatility
    assert "method='simple'" in issues[1].message  # forward_return applies V10 the same way


# --- P01 PLANNER_REFUSED --------------------------------------------------------


def test_p01_refusal(catalog: Catalog) -> None:
    report = validate(raw_plan("arima_forecast_block"), catalog)
    assert report.codes() == ["P01"] and report.is_blocked


def test_v02_blocks_a_planner_that_follows_an_injection(catalog: Catalog) -> None:
    # Scenario 4 simulates a planner that obeys the injection: a well-formed plan
    # whose second step names a capability that does not exist. Only V02 fires.
    report = validate(raw_plan("prompt_injection_unknown_capability"), catalog)
    assert [(i.issue_id, i.action) for i in report.issues] == [("V02:s2", "BLOCK")]
    assert "'run_python' is not a registered capability" in report.issues[0].message


# --- V11 FORWARD_DATA_AS_SIGNAL ---------------------------------------------------


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
def test_v11_forward_returns_are_only_an_outcome(
    capability: str, inputs: dict[str, str], params: dict[str, Any], catalog: Catalog
) -> None:
    raw = raw_plan(SUCCESS)
    raw["steps"].append({"id": "s9", "capability": capability, "inputs": inputs,
                         "params": params})  # fmt: skip
    report = validate(raw, catalog)
    forward_input = next(name for name, ref in inputs.items() if ref == "s6")
    assert [i.issue_id for i in report.blocks] == [f"V11:s9:{forward_input}"]
    assert "can only be used as an outcome, never to build a signal" in report.blocks[0].message
    assert "V04" not in report.codes()  # the specific guardrail replaces the generic one


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


# The module that raises the issues of each code prefix. guardrails.py holds the
# request check (R, called by the engine before the planner) and the validator's V
# codes, and turns the planner's refusal (P01) into a BLOCK, so a refused plan can
# never run; the executor raises the E codes and the narrator check the N codes. In
# particular, no executor check can be labelled R.
RAISED_IN = {
    "R": "guardrails.py",
    "P": "guardrails.py",
    "V": "guardrails.py",
    "E": "executor.py",
    "N": "narrator.py",
}


def test_guardrail_table_matches_the_implemented_checks() -> None:
    raised = make_issue_codes()
    # Every code in the table is raised somewhere, and nothing raised is undescribed.
    assert {code for _, code in raised} == set(GUARDRAILS)
    for file, code in raised:
        assert file == RAISED_IN[code[0]], f"{code} is raised in {file}"


def test_each_code_prefix_names_its_stage_and_type() -> None:
    # In pipeline order: request check, planner, validator, executor, narrator check.
    assert list(ENFORCERS.items()) == [
        ("R", "Request check"),
        ("P", "Planner"),
        ("V", "Validator"),
        ("E", "Executor"),
        ("N", "Narrator check"),
    ]
    assert sorted(LLM_STAGES) == ["Planner"]  # every other stage is deterministic code
    stage_and_type = {code: (g.enforced_by, g.type) for code, g in GUARDRAILS.items()}
    deterministic = GuardrailType.DETERMINISTIC
    assert stage_and_type == {
        "R01": ("Request check", deterministic),
        "P01": ("Planner", GuardrailType.LLM),
        **{f"V{i:02d}": ("Validator", deterministic) for i in range(1, 12)},
        **{f"E{i:02d}": ("Executor", deterministic) for i in range(1, 6)},
        "N01": ("Narrator check", deterministic),
    }
    assert list(GUARDRAILS) == list(stage_and_type)  # the table is in pipeline order
    for code, guardrail in GUARDRAILS.items():
        assert guardrail.code == code and re.fullmatch(r"[RPVEN]\d{2}", code)
    # An issue reads its stage, type and category from the table, by code, and
    # writes them into the run record (computed fields), which ignores them on load.
    issue = planner_failure_issue("timeout")
    assert (issue.enforced_by, issue.type, issue.category) == (
        "Validator", "Deterministic", "Plan integrity")  # fmt: skip
    dumped = issue.model_dump(mode="json")
    assert (dumped["enforced_by"], dumped["type"], dumped["category"]) == (
        "Validator", "Deterministic", "Plan integrity")  # fmt: skip
    assert Issue.model_validate(dumped) == issue


# Every guardrail code assigned since the codes were named by pipeline stage
# (docs/DECISIONS.md lists the three earlier numberings). From then on codes are
# permanent: a code is never renumbered or given another meaning. Add a new
# guardrail's code here; when a guardrail is retired, move its entry to
# RETIRED_GUARDRAIL_CODES.
ASSIGNED_GUARDRAIL_CODES = {
    "R01": "REQUEST_TOO_LONG",
    "P01": "PLANNER_REFUSED",
    "V01": "INVALID_PLAN",
    "V02": "UNKNOWN_CAPABILITY",
    "V03": "INVALID_PARAM",
    "V04": "TYPE_MISMATCH",
    "V05": "BUDGET_EXCEEDED",
    "V06": "UNKNOWN_DATASET",
    "V07": "RETURNS_ON_RATES",
    "V08": "FREQUENCY_MISMATCH",
    "V09": "LOOK_AHEAD",
    "V10": "MATERIAL_ASSUMPTION",
    "V11": "FORWARD_DATA_AS_SIGNAL",
    "E01": "DATASET_HASH_MISMATCH",
    "E02": "INSUFFICIENT_DATA",
    "E03": "MISSING_DATA",
    "E04": "ALIGNMENT_LOSS",
    "E05": "LOW_EFFECTIVE_SAMPLE",
    "N01": "UNGROUNDED_NARRATIVE",
}
RETIRED_GUARDRAIL_CODES: dict[str, str] = {}


def test_guardrail_codes_are_never_renumbered_or_reused() -> None:
    current = {code: g.name for code, g in GUARDRAILS.items()}
    for code, name in ASSIGNED_GUARDRAIL_CODES.items():
        assert current.get(code) == name, f"{code} was assigned to {name}"
    for code, name in RETIRED_GUARDRAIL_CODES.items():
        assert code not in current, f"{code} belonged to the retired {name}"
    unrecorded = sorted(set(current) - set(ASSIGNED_GUARDRAIL_CODES))
    assert unrecorded == [], f"record new codes in ASSIGNED_GUARDRAIL_CODES: {unrecorded}"


def test_descriptions_quote_the_runtime_thresholds() -> None:
    # The descriptions are plain text shown in the UI; they must state the
    # thresholds the executor actually applies.
    assert f"more than {MISSING_DATA_THRESHOLD:.0%}" in GUARDRAILS["E03"].description
    assert f"more than {ALIGNMENT_LOSS_THRESHOLD:.0%}" in GUARDRAILS["E04"].description
    assert f"fewer than {MIN_FLAGGED_EPISODES} " in GUARDRAILS["E05"].description


def test_v05_and_r01_descriptions_quote_their_limits() -> None:
    defaults = {f.name: f.default for f in dataclasses.fields(Settings)}
    assert f"at most {defaults['max_steps']} steps by default" in GUARDRAILS["V05"].description
    assert "request" not in GUARDRAILS["V05"].description  # plan limits only
    assert (f"over {MAX_REQUEST_WORDS:,} words, or {MAX_REQUEST_CHARS:,} characters"
            in GUARDRAILS["R01"].description)  # fmt: skip


def test_every_guardrail_has_an_action_a_category_and_a_description() -> None:
    for guardrail in GUARDRAILS.values():
        assert guardrail.action in Action and guardrail.category.value
        assert guardrail.description.endswith("."), guardrail.code


# The audited category of every guardrail (docs/DECISIONS.md): one each. R01 checks the
# request and E01 the data file, both inputs, so they are Input validity.
AUDITED_CATEGORIES = {
    Category.PLAN_INTEGRITY: ["V01", "V05"],
    Category.APPROVED_SCOPE: ["P01", "V02", "V06"],
    Category.INPUT_VALIDITY: ["R01", "V03", "V04", "E01"],
    Category.METHODOLOGY: ["V07", "V08", "V09", "V10", "V11", "E05"],
    Category.DATA_QUALITY: ["E02", "E03", "E04"],
    Category.OUTPUT_GROUNDING: ["N01"],
}


def test_every_guardrail_has_its_audited_category() -> None:
    actual = {c: [code for code, g in GUARDRAILS.items() if g.category == c] for c in Category}
    assert actual == AUDITED_CATEGORIES


def test_every_category_is_defined_and_used() -> None:
    # One list (the enum), one definition each, and no empty category.
    assert list(CATEGORY_DEFINITIONS) == list(Category)
    assert all(d.endswith(".") for d in CATEGORY_DEFINITIONS.values())
    assert {g.category for g in GUARDRAILS.values()} == set(Category)


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
