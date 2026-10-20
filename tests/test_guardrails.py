"""The GUARDRAILS table and the request check (R01).

Checks that the table (shown on the Guardrails page) lists exactly the implemented
checks, that each code's prefix names who enforces it and the module that raises it,
that codes are never renumbered, that only V01 and V02 are terminal, and that the
descriptions quote the executor's thresholds; and that the request check blocks an
oversized request and passes a normal one. The validator's checks are tested in
test_validator.py.
"""

import ast
import dataclasses
import re

import pytest

from quantlab.config import PROJECT_ROOT, Settings
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
    check_request_length,
)
from quantlab.mock_scenarios import VIX_CHF_REQUEST
from quantlab.validator import planner_failure_issue

# --- R01 REQUEST_TOO_LONG (the request check, before the planner) ----------------


def test_r01_passes_a_normal_request_and_one_at_the_limits() -> None:
    assert check_request_length(VIX_CHF_REQUEST) == []
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
# request check (R, called by the graph before the planner); validator.py raises the
# V codes and turns the planner's refusal (P01) into a BLOCK, so a refused plan can
# never run; the executor raises the E codes and the narrator check the N codes. In
# particular, no executor check can be labelled R.
RAISED_IN = {
    "R": "guardrails.py",
    "P": "validator.py",
    "V": "validator.py",
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


def test_only_v01_and_v02_are_terminal() -> None:
    terminal = {code for code, g in GUARDRAILS.items() if g.fixable_by_edit is False}
    assert terminal == {"V01", "V02"}
    for code, g in GUARDRAILS.items():
        if g.action != Action.BLOCK or code in ("R01", "P01"):  # no plan to edit
            assert g.fixable_by_edit is None and g.fixable_text == "n/a", code
        else:
            assert isinstance(g.fixable_by_edit, bool), code


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
