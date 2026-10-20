"""Guardrails: the table of every guardrail, the shared issue types and the request check.

Role in the architecture: the catalogue every stage shares. Every guardrail,
whoever enforces it, is described in the single GUARDRAILS table below, and every
problem a guardrail finds is an Issue, built by `make_issue` from that table. Each
issue says what happens next, its action:

- BLOCK: the run stops (unknown capability, bad parameter, returns of a yield...).
- APPROVE: a human must decide; the plan runs only after they approve the issue
  (look-ahead thresholds, material modelling assumptions).
- WARN: the run continues and the result is flagged: the warning is shown, passed
  to the narrator and kept in the run record.

The first letter of a guardrail's code names the pipeline stage that enforces it,
in pipeline order (ENFORCERS), and each stage raises its own codes:
    R  the request check (deterministic), before the planner: this module, called
       by the graph's request_check node
    P  the planner (LLM): it refuses, and validator.py turns the refusal into a BLOCK
    V  the validator (deterministic), on the plan before anything runs: validator.py
    E  the executor, at run time on the data: executor.py
    N  the narrator check (deterministic), on the written summary: narrator.py
Human approval is not a guardrail: it is the step where APPROVE issues are resolved.
The codes were renumbered three times while the design settled (docs/DECISIONS.md).
From this version on they are permanent: a code is never renumbered or given a
different meaning, so the issue ids recorded in run records keep their meaning.

This module imports no other stage, so every stage can import it: validator.py
imports it, never the other way round. The request check (R01) is a stage of its
own, too small for a module of its own, so it is written here.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, computed_field

from quantlab.types import Frequency


class Action(StrEnum):
    """What happens when a guardrail fires."""

    BLOCK = "BLOCK"  # the run stops
    APPROVE = "APPROVE"  # a human must decide: the plan runs only after explicit approval
    WARN = "WARN"  # the run continues; the result is flagged


class Category(StrEnum):
    """What a guardrail protects (CATEGORY_DEFINITIONS says what each one covers)."""

    PLAN_INTEGRITY = "Plan integrity"
    APPROVED_SCOPE = "Approved scope"
    INPUT_VALIDITY = "Input validity"
    METHODOLOGY = "Methodology"
    DATA_QUALITY = "Data quality"
    OUTPUT_GROUNDING = "Output grounding"


# What each category covers, in one line: the only place the definitions are written,
# read by the Guardrails page (the Category filter's tooltip).
CATEGORY_DEFINITIONS = {
    Category.PLAN_INTEGRITY: "The plan is well-formed and of a size a person can review.",
    Category.APPROVED_SCOPE: (
        "Only approved capabilities and catalogued datasets, for requests the registry can serve."
    ),
    Category.INPUT_VALIDITY: (
        "Every input is acceptable: the request's size, each parameter, each step's input "
        "kind, and data files that match their recorded hash."
    ),
    Category.METHODOLOGY: "The analysis is statistically and economically sound.",
    Category.DATA_QUALITY: (
        "There is enough data, and gaps or alignment do not silently distort it."
    ),
    Category.OUTPUT_GROUNDING: "The written summary quotes only values the run computed or used.",
}


class GuardrailType(StrEnum):
    """Whether a guardrail is enforced by deterministic code or by an LLM."""

    DETERMINISTIC = "Deterministic"
    LLM = "LLM"


# The pipeline stage that enforces a guardrail, by the first letter of its code, in
# pipeline order, and the stages that are an LLM (every other stage is deterministic
# code). This is the only place either is written down, so a code, its stage and its
# type can never disagree.
ENFORCERS = {
    "R": "Request check",
    "P": "Planner",
    "V": "Validator",
    "E": "Executor",
    "N": "Narrator check",
}
LLM_STAGES = {"Planner"}

# R01 REQUEST_TOO_LONG: the longest request sent to the planner. Words are what a
# person writes; the character cap (about 1,000 words at 6 characters each, with a
# margin) stops a text without spaces, which counts as one "word", slipping through.
MAX_REQUEST_WORDS = 1000
MAX_REQUEST_CHARS = 8000


@dataclass(frozen=True)
class Guardrail:
    """Description of one guardrail (used for issues, the UI and docs/ARCHITECTURE.md).

    Fields:
        code: e.g. "V07"; its first letter names the stage that enforces it, which
            also gives its type (see ENFORCERS and LLM_STAGES).
        name: e.g. "RETURNS_ON_RATES".
        action: what happens when it fires: BLOCK, APPROVE or WARN.
        category: what it protects.
        description: what it checks, and what that prevents.
        fixable_by_edit: for a BLOCK on a plan, whether editing parameter values can
            fix it. False makes it terminal: the whole plan is rejected and can be
            neither edited, re-validated, approved nor executed (its steps are wrong in
            kind or structure, which value edits cannot change). None where it does not
            apply: APPROVE and WARN, and R01 and P01, which leave no plan to edit.
    """

    code: str
    name: str
    action: Action
    category: Category
    description: str
    fixable_by_edit: bool | None = None

    @property
    def fixable_text(self) -> str:
        """fixable_by_edit as the Guardrails page and docs/ARCHITECTURE.md show it:
        "yes", "no" (terminal: the plan is rejected) or "n/a"."""
        return {True: "yes", False: "no", None: "n/a"}[self.fixable_by_edit]

    @property
    def enforced_by(self) -> str:
        """The pipeline stage that enforces it, e.g. "Validator" for a V code."""
        return ENFORCERS[self.code[0]]

    @property
    def type(self) -> GuardrailType:
        """LLM if its stage is an LLM (the planner), Deterministic otherwise."""
        if self.enforced_by in LLM_STAGES:
            return GuardrailType.LLM
        return GuardrailType.DETERMINISTIC


BLOCK, APPROVE, WARN = Action.BLOCK, Action.APPROVE, Action.WARN

# Plain-English names used in messages, and an order (higher = less frequent).
FREQUENCY_NAMES = {Frequency.D: "daily", Frequency.W: "weekly", Frequency.M: "monthly"}
PERIOD_NAMES = {Frequency.D: "day", Frequency.W: "week", Frequency.M: "month"}
FREQUENCY_RANK = {Frequency.D: 0, Frequency.W: 1, Frequency.M: 2}

# Every guardrail, in pipeline order: request check, planner, validator, executor,
# narrator check.
# The descriptions are shown as they are on the Guardrails page and in docs/ARCHITECTURE.md.
GUARDRAILS: dict[str, Guardrail] = {
    g.code: g
    for g in [
        Guardrail(
            "R01", "REQUEST_TOO_LONG", BLOCK, Category.INPUT_VALIDITY,
            "Blocks a request over 1,000 words, or 8,000 characters (so a text without "
            "spaces cannot slip through), before it reaches the LLM. Prevents pasted "
            "documents or stuffed instructions from driving the planner.",
        ),
        Guardrail(
            "P01", "PLANNER_REFUSED", BLOCK, Category.APPROVED_SCOPE,
            "The planner may decline a request it cannot serve with the approved "
            "capabilities, or one that tries to override its rules, giving a reason instead "
            "of a plan (in mock mode, any request without a canned plan); nothing runs. "
            "Prevents an out-of-scope or malicious request from being forced into a "
            "misleading plan.",
        ),
        Guardrail(
            "V01", "INVALID_PLAN", BLOCK, Category.PLAN_INTEGRITY,
            "The plan must match the plan schema exactly (no missing or extra fields), have "
            "at least one step, use unique step ids, and give each step exactly the inputs "
            "its capability defines, each taken from an earlier step. Also raised when the "
            "planner returns no usable plan: an API error or timeout, an answer cut off or "
            "refused by the provider, or no readable plan submitted. Prevents a malformed, "
            "ambiguous or out-of-order plan from running.",
            fixable_by_edit=False,
        ),
        Guardrail(
            "V02", "UNKNOWN_CAPABILITY", BLOCK, Category.APPROVED_SCOPE,
            "Every step must call a capability in the approved registry, so no unreviewed or "
            "untested code is ever run, whatever the LLM asks for.",
            fixable_by_edit=False,
        ),
        Guardrail(
            "V03", "INVALID_PARAM", BLOCK, Category.INPUT_VALIDITY,
            "Each parameter must be one its capability defines, given if required, of the "
            "right type and within its allowed range or choices (for example a quantile "
            "between 0.5 and 0.99). User edits that change anything other than a parameter "
            "value, such as a step, an input or the capability, are rejected too. Prevents "
            "nonsensical settings from producing plausible-looking numbers.",
            fixable_by_edit=True,
        ),
        Guardrail(
            "V04", "TYPE_MISMATCH", BLOCK, Category.INPUT_VALIDITY,
            "Each input must be the data type and kind of series its capability accepts: for "
            "example returns, not prices, for a volatility, and returns, not a price or rate "
            "level, for the significance test. Prevents statistically meaningless "
            "calculations.",
            fixable_by_edit=True,
        ),
        Guardrail(
            "V05", "BUDGET_EXCEEDED", BLOCK, Category.PLAN_INTEGRITY,
            "The plan may have at most 15 steps by default (QUANTLAB_MAX_STEPS). Prevents "
            "runaway plans too long for a human to review.",
            fixable_by_edit=True,
        ),
        Guardrail(
            "V06", "UNKNOWN_DATASET", BLOCK, Category.APPROVED_SCOPE,
            "Every dataset the plan names or loads must be in the approved catalog. Prevents "
            "analysis of invented or unapproved data.",
            fixable_by_edit=True,
        ),
        Guardrail(
            "V07", "RETURNS_ON_RATES", BLOCK, Category.METHODOLOGY,
            "Returns cannot be computed on an interest rate or yield: rates are differenced "
            "instead (changes in percentage points). Prevents meaningless results, such as a "
            "move from 1% to 2% reported as +100%.",
            fixable_by_edit=True,
        ),
        Guardrail(
            "V08", "FREQUENCY_MISMATCH", BLOCK, Category.METHODOLOGY,
            "Series combined in one step (a spread, a comparison or the significance test) "
            "must have the same frequency. Prevents silently matching weekly values with "
            "daily ones on the few dates they share.",
            fixable_by_edit=True,
        ),
        Guardrail(
            "V09", "LOOK_AHEAD", APPROVE, Category.METHODOLOGY,
            "A flag threshold computed from the full sample, which includes data from after "
            "each flagged date, needs explicit approval. Prevents results that look better "
            "than anything achievable in real time from passing unnoticed.",
        ),
        Guardrail(
            "V10", "MATERIAL_ASSUMPTION", APPROVE, Category.METHODOLOGY,
            "Choices that change the answer need explicit approval: today, the return method "
            "(log or simple) of compute_returns and forward_return. Prevents the LLM from "
            "making material assumptions silently.",
        ),
        Guardrail(
            "V11", "FORWARD_DATA_AS_SIGNAL", BLOCK, Category.METHODOLOGY,
            "Forward returns may only be the outcome of a comparison or of the significance "
            "test, never the input to a volatility, a flag or any other step. Prevents future "
            "information from leaking into a signal.",
            fixable_by_edit=True,
        ),
        Guardrail(
            "E01", "DATASET_HASH_MISMATCH", BLOCK, Category.INPUT_VALIDITY,
            "Each snapshot file must match the hash recorded in the catalog before it is "
            "loaded. Prevents analysis of altered or different data.",
            fixable_by_edit=True,
        ),
        Guardrail(
            "E02", "INSUFFICIENT_DATA", BLOCK, Category.DATA_QUALITY,
            "Each step needs its capability's minimum number of observations, for example "
            "three times the window for a rolling volatility and 30 shared dates for the "
            "significance test. Prevents statistics computed on too little data.",
            fixable_by_edit=True,
        ),
        Guardrail(
            "E03", "MISSING_DATA", WARN, Category.DATA_QUALITY,
            "Warns when more than 5% of a loaded series' values are missing (they are "
            "dropped). Prevents gaps from silently distorting the results.",
        ),
        Guardrail(
            "E04", "ALIGNMENT_LOSS", WARN, Category.DATA_QUALITY,
            "Warns when aligning two series on their common dates drops more than 10% of "
            "either. Prevents a silently shrunken sample.",
        ),
        Guardrail(
            "E05", "LOW_EFFECTIVE_SAMPLE", WARN, Category.METHODOLOGY,
            "Warns when the flagged periods of a comparison or of the significance test form "
            "fewer than 5 distinct episodes (runs of consecutive flags). Prevents "
            "overconfidence in a result that rests on a handful of real events.",
        ),
        Guardrail(
            "N01", "UNGROUNDED_NARRATIVE", WARN, Category.OUTPUT_GROUNDING,
            "Every number in the written summary must be a correct rounding of a value the "
            "run computed or used: results, parameters, the request, warnings, dataset "
            "titles, annualisation factors and, when a test ran, the 5% and 95% levels. "
            "Dates must be dates of the results (a year on its own, within their range). A "
            "failing LLM summary is withheld and the deterministic one, checked the same "
            "way, is shown instead, with the failing numbers listed. Prevents invented or "
            "misquoted figures.",
        ),
    ]
}  # fmt: skip

# The input that holds the OUTCOME in the capabilities that compare an outcome
# between flagged and normal periods. It is the only place forward returns may
# go (V11), and where E05 counts the flagged episodes.
OUTCOME_INPUTS = {"compare_groups": "values", "hac_mean_difference": "outcome"}


class Issue(BaseModel):
    """One problem found by a guardrail.

    Fields:
        code: guardrail code, e.g. "V07".
        name: guardrail name, e.g. "RETURNS_ON_RATES".
        action: BLOCK, APPROVE or WARN.
        step_id: the step concerned (None for plan-level issues).
        message: what is wrong.
        suggestion: how to fix it.
        issue_id: stable id used to record approvals, e.g. "V10:s2:return_method".

    Who enforced it, its type and what it protects are not stored: enforced_by,
    type and category read them from the GUARDRAILS table by code. They are
    computed fields, so they are written into run records, for whoever reads a
    manifest, and ignored when a record is loaded.
    """

    code: str
    name: str
    action: Action
    step_id: str | None
    message: str
    suggestion: str | None = None
    issue_id: str

    @computed_field
    @property
    def enforced_by(self) -> str:
        """The stage that enforced it, e.g. "Executor" (from the GUARDRAILS table)."""
        return GUARDRAILS[self.code].enforced_by

    @computed_field
    @property
    def type(self) -> GuardrailType:
        """Deterministic or LLM (from the GUARDRAILS table)."""
        return GUARDRAILS[self.code].type

    @computed_field
    @property
    def category(self) -> Category:
        """What it protects, e.g. "Methodology" (from the GUARDRAILS table)."""
        return GUARDRAILS[self.code].category


def make_issue(
    code: str,
    step_id: str | None,
    message: str,
    suggestion: str | None = None,
    key: str | None = None,
) -> Issue:
    """Create an Issue, taking its name and action from the GUARDRAILS table.

    Args:
        code: guardrail code.
        step_id: step concerned, or None.
        message: what is wrong.
        suggestion: how to fix it.
        key: extra discriminator for the issue_id when one step can raise the
            same code more than once (e.g. the parameter or assumption name).

    Returns:
        The Issue.
    """
    guardrail = GUARDRAILS[code]
    issue_id = ":".join(part for part in (code, step_id or "plan", key) if part)
    return Issue(
        code=code,
        name=guardrail.name,
        action=guardrail.action,
        step_id=step_id,
        message=message,
        suggestion=suggestion,
        issue_id=issue_id,
    )


class ValidationReport(BaseModel):
    """The result of validating a plan.

    Fields:
        issues: every issue found, in plan order.
        dry_run: step id -> predicted output metadata (data_type, kind, frequency...).
    """

    issues: list[Issue] = Field(default_factory=list)
    dry_run: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @property
    def blocks(self) -> list[Issue]:
        """Issues that stop the plan from running."""
        return [i for i in self.issues if i.action == Action.BLOCK]

    @property
    def is_blocked(self) -> bool:
        """True if any BLOCK issue exists."""
        return bool(self.blocks)

    @property
    def rejections(self) -> list[Issue]:
        """BLOCK issues no edit can fix (fixable_by_edit False, e.g. V02 for every
        unregistered step): the whole plan is rejected."""
        return [i for i in self.blocks if GUARDRAILS[i.code].fixable_by_edit is False]

    @property
    def is_rejected(self) -> bool:
        """True if the plan is rejected as a whole: only a new request can go on."""
        return bool(self.rejections)

    @property
    def approvals_required(self) -> list[Issue]:
        """Issues that need explicit user approval before execution."""
        return [i for i in self.issues if i.action == Action.APPROVE]

    @property
    def warnings(self) -> list[Issue]:
        """Non-blocking warnings."""
        return [i for i in self.issues if i.action == Action.WARN]

    def codes(self) -> list[str]:
        """Sorted, de-duplicated issue codes, e.g. ["V02", "V07"]."""
        return sorted({i.code for i in self.issues})


# ---------------------------------------------------------------------------
# Request check (R01): before the planner
# ---------------------------------------------------------------------------


def check_request_length(request: str) -> list[Issue]:
    """R01 REQUEST_TOO_LONG: refuse a request over MAX_REQUEST_WORDS words or MAX_REQUEST_CHARS.

    The graph (and the evaluation) call it before the planner, so an oversized
    request costs no LLM call. Without it, a pasted document could blow up prompt
    size and cost, and gives room for instructions stuffed inside it to steer the
    planner. Words are counted as whitespace-separated pieces of text.

    Args:
        request: the user's request.

    Returns:
        One BLOCK issue if either limit is exceeded, else [].
    """
    n_words, n_chars = len(request.split()), len(request)
    if n_words <= MAX_REQUEST_WORDS and n_chars <= MAX_REQUEST_CHARS:
        return []
    return [
        make_issue(
            "R01",
            None,
            f"The request has {n_words:,} words and {n_chars:,} characters; the limits are "
            f"{MAX_REQUEST_WORDS:,} words and {MAX_REQUEST_CHARS:,} characters.",
            "Shorten the request to the question itself; paste no documents into it.",
            key="request",
        )
    ]
