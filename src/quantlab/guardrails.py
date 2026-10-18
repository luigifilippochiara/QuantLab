"""Guardrails: the table of every guardrail, and the validator's static checks.

Role in the architecture: the validator is the "system disposes" half of
"the LLM proposes, the system disposes". It takes a plan (inert data) and,
without touching any data values, decides whether it may run. Every issue a
guardrail raises says what happens next, its action:

- BLOCK: the run stops (unknown capability, bad parameter, returns of a yield...).
- APPROVE: a human must decide; the plan runs only after they approve the issue
  (look-ahead thresholds, material modelling assumptions).
- WARN: the run continues and the result is flagged: the warning is shown, passed
  to the narrator and kept in the run record.

It runs a metadata "dry run": walking the steps in order and propagating each
output's data type, kind and frequency using only the registry and the catalog.
That is how it can tell, before running, that step 3 would compute returns of
a yield, take the spread of a daily and a weekly series, or build a flag from
forward returns.

The same `validate_plan` function is used for the planner's plan, for every
user edit, by the executor (defence in depth) and by replay. `check_edits`
locks the plan's structure: users may change parameter values only.

Every guardrail, whoever enforces it, is described in the single GUARDRAILS
table below. The first letter of its code says who enforces it (ENFORCERS):
    L  the planner (LLM): it refuses, and the validator turns the refusal into a BLOCK
    G  the validator (deterministic): this module
    R  the executor, at run time on the data: executor.py
    N  the narrator check (deterministic): narrator.py
The codes were renumbered twice while the design settled (docs/DECISIONS.md).
From this version on they are permanent: a code is never renumbered or given a
different meaning, so the issue ids recorded in run records keep their meaning.
"""

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from quantlab.capabilities import FORWARD_HORIZON
from quantlab.data import Catalog
from quantlab.plan import Plan, Step
from quantlab.registry import FROM_CATALOG, REGISTRY, SAME_AS_INPUT, CapabilitySpec, ParamSpec
from quantlab.types import DataType, Frequency, SeriesKind


class Action(StrEnum):
    """What happens when a guardrail fires."""

    BLOCK = "BLOCK"  # the run stops
    APPROVE = "APPROVE"  # a human must decide: the plan runs only after explicit approval
    WARN = "WARN"  # the run continues; the result is flagged


class Category(StrEnum):
    """What a guardrail protects."""

    PLAN_INTEGRITY = "Plan integrity"  # the plan is well-formed and small enough to review
    APPROVED_SCOPE = "Approved scope"  # only registered capabilities and catalog data
    INPUT_VALIDITY = "Input validity"  # parameters and input kinds make sense
    METHODOLOGY = "Methodology"  # the analysis is statistically and economically sound
    DATA_QUALITY = "Data quality"  # the data is the recorded version, and there is enough
    OUTPUT_GROUNDING = "Output grounding"  # the narrative quotes only values of the run


# Who enforces a guardrail, by the first letter of its code. This is the only
# place it is written down, so a code and its label can never disagree.
ENFORCERS = {
    "L": "Planner (LLM)",
    "G": "Validator (deterministic)",
    "R": "Executor (runtime)",
    "N": "Narrator check (deterministic)",
}


@dataclass(frozen=True)
class Guardrail:
    """Description of one guardrail (used for issues, the UI and docs/ARCHITECTURE.md).

    Fields:
        code: e.g. "G07"; its first letter says who enforces it (see ENFORCERS).
        name: e.g. "RETURNS_ON_RATES".
        action: what happens when it fires: BLOCK, APPROVE or WARN.
        category: what it protects.
        description: what it checks, and what that prevents.
    """

    code: str
    name: str
    action: Action
    category: Category
    description: str

    @property
    def enforced_by(self) -> str:
        """Who enforces it, e.g. "Validator (deterministic)" for a G code."""
        return ENFORCERS[self.code[0]]


BLOCK, APPROVE, WARN = Action.BLOCK, Action.APPROVE, Action.WARN

# Plain-English names used in messages, and an order (higher = less frequent).
FREQUENCY_NAMES = {Frequency.D: "daily", Frequency.W: "weekly", Frequency.M: "monthly"}
PERIOD_NAMES = {Frequency.D: "day", Frequency.W: "week", Frequency.M: "month"}
FREQUENCY_RANK = {Frequency.D: 0, Frequency.W: 1, Frequency.M: 2}

# Every guardrail, in pipeline order: planner, validator, executor, narrator check.
# The descriptions are shown as they are on the Guardrails page and in docs/ARCHITECTURE.md.
GUARDRAILS: dict[str, Guardrail] = {
    g.code: g
    for g in [
        Guardrail(
            "L01", "PLANNER_REFUSED", BLOCK, Category.APPROVED_SCOPE,
            "The planner may decline a request it cannot serve with the approved "
            "capabilities, or one that tries to override its rules, giving a reason instead "
            "of a plan (in mock mode, any request without a canned plan); nothing runs. "
            "Prevents an out-of-scope or malicious request from being forced into a "
            "misleading plan.",
        ),
        Guardrail(
            "G01", "INVALID_PLAN", BLOCK, Category.PLAN_INTEGRITY,
            "The plan must match the plan schema exactly (no missing or extra fields), have "
            "at least one step, use unique step ids, and give each step exactly the inputs "
            "its capability defines, each taken from an earlier step. Also raised when the "
            "planner returns no usable plan: an API error or timeout, an answer cut off or "
            "refused by the provider, or no readable plan submitted. Prevents a malformed, "
            "ambiguous or out-of-order plan from running.",
        ),
        Guardrail(
            "G02", "UNKNOWN_CAPABILITY", BLOCK, Category.APPROVED_SCOPE,
            "Every step must call a capability in the approved registry, so no unreviewed or "
            "untested code is ever run, whatever the LLM asks for.",
        ),
        Guardrail(
            "G03", "INVALID_PARAM", BLOCK, Category.INPUT_VALIDITY,
            "Each parameter must be one its capability defines, given if required, of the "
            "right type and within its allowed range or choices (for example a quantile "
            "between 0.5 and 0.99). User edits that change anything other than a parameter "
            "value, such as a step, an input or the capability, are rejected too. Prevents "
            "nonsensical settings from producing plausible-looking numbers.",
        ),
        Guardrail(
            "G04", "TYPE_MISMATCH", BLOCK, Category.INPUT_VALIDITY,
            "Each input must be the data type and kind of series its capability accepts: for "
            "example returns, not prices, for a volatility, and returns, not a price or rate "
            "level, for the significance test. Prevents statistically meaningless "
            "calculations.",
        ),
        Guardrail(
            "G05", "BUDGET_EXCEEDED", BLOCK, Category.PLAN_INTEGRITY,
            "The request must be at most 2,000 characters, checked before the planner is "
            "called (so an oversized request costs no LLM call) and again with every plan; "
            "the plan may have at most 15 steps by default (QUANTLAB_MAX_STEPS). Prevents "
            "oversized prompts and runaway plans too long for a human to review.",
        ),
        Guardrail(
            "G06", "UNKNOWN_DATASET", BLOCK, Category.APPROVED_SCOPE,
            "Every dataset the plan names or loads must be in the approved catalog. Prevents "
            "analysis of invented or unapproved data.",
        ),
        Guardrail(
            "G07", "RETURNS_ON_RATES", BLOCK, Category.METHODOLOGY,
            "Returns cannot be computed on an interest rate or yield: rates are differenced "
            "instead (changes in percentage points). Prevents meaningless results, such as a "
            "move from 1% to 2% reported as +100%.",
        ),
        Guardrail(
            "G08", "FREQUENCY_MISMATCH", BLOCK, Category.METHODOLOGY,
            "Series combined in one step (a spread, a comparison or the significance test) "
            "must have the same frequency. Prevents silently matching weekly values with "
            "daily ones on the few dates they share.",
        ),
        Guardrail(
            "G09", "LOOK_AHEAD", APPROVE, Category.METHODOLOGY,
            "A flag threshold computed from the full sample, which includes data from after "
            "each flagged date, needs explicit approval. Prevents results that look better "
            "than anything achievable in real time from passing unnoticed.",
        ),
        Guardrail(
            "G10", "MATERIAL_ASSUMPTION", APPROVE, Category.METHODOLOGY,
            "Choices that change the answer need explicit approval: today, the return method "
            "(log or simple) of compute_returns and forward_return. Prevents the LLM from "
            "making material assumptions silently.",
        ),
        Guardrail(
            "G11", "FORWARD_DATA_AS_SIGNAL", BLOCK, Category.METHODOLOGY,
            "Forward returns may only be the outcome of a comparison or of the significance "
            "test, never the input to a volatility, a flag or any other step. Prevents future "
            "information from leaking into a signal.",
        ),
        Guardrail(
            "R01", "DATASET_HASH_MISMATCH", BLOCK, Category.DATA_QUALITY,
            "Each snapshot file must match the hash recorded in the catalog before it is "
            "loaded. Prevents analysis of altered or different data.",
        ),
        Guardrail(
            "R02", "INSUFFICIENT_DATA", BLOCK, Category.DATA_QUALITY,
            "Each step needs its capability's minimum number of observations, for example "
            "three times the window for a rolling volatility and 30 shared dates for the "
            "significance test. Prevents statistics computed on too little data.",
        ),
        Guardrail(
            "R03", "MISSING_DATA", WARN, Category.DATA_QUALITY,
            "Warns when more than 5% of a loaded series' values are missing (they are "
            "dropped). Prevents gaps from silently distorting the results.",
        ),
        Guardrail(
            "R04", "ALIGNMENT_LOSS", WARN, Category.DATA_QUALITY,
            "Warns when aligning two series on their common dates drops more than 10% of "
            "either. Prevents a silently shrunken sample.",
        ),
        Guardrail(
            "R05", "LOW_EFFECTIVE_SAMPLE", WARN, Category.METHODOLOGY,
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
# go (G11), and where R05 counts the flagged episodes.
OUTCOME_INPUTS = {"compare_groups": "values", "hac_mean_difference": "outcome"}


class Issue(BaseModel):
    """One problem found by a guardrail.

    Fields:
        code: guardrail code, e.g. "G07".
        name: guardrail name, e.g. "RETURNS_ON_RATES".
        action: BLOCK, APPROVE or WARN.
        step_id: the step concerned (None for plan-level issues).
        message: what is wrong.
        suggestion: how to fix it.
        issue_id: stable id used to record approvals, e.g. "G10:s2:return_method".

    Who enforced it and what it protects are not stored: the enforced_by and
    category properties read them from the GUARDRAILS table by code.
    """

    code: str
    name: str
    action: Action
    step_id: str | None
    message: str
    suggestion: str | None = None
    issue_id: str

    @property
    def enforced_by(self) -> str:
        """Who enforced it, e.g. "Executor (runtime)" (from the GUARDRAILS table)."""
        return GUARDRAILS[self.code].enforced_by

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
    def approvals_required(self) -> list[Issue]:
        """Issues that need explicit user approval before execution."""
        return [i for i in self.issues if i.action == Action.APPROVE]

    @property
    def warnings(self) -> list[Issue]:
        """Non-blocking warnings."""
        return [i for i in self.issues if i.action == Action.WARN]

    def codes(self) -> list[str]:
        """Sorted, de-duplicated issue codes, e.g. ["G02", "G07"]."""
        return sorted({i.code for i in self.issues})


@dataclass
class StaticMeta:
    """Output metadata predicted by the dry run (no data values involved).

    Fields mirror ArtifactMeta: data type, kind, frequency, return method,
    return/difference periods, forward-return horizon and the source dataset ids.
    """

    data_type: DataType
    kind: SeriesKind | None
    frequency: Frequency | None
    return_method: str | None = None
    periods: int | None = None
    horizon: int | None = None
    dataset_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict version for reports and the UI."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------


def effective_params(step: Step, spec: CapabilitySpec) -> dict[str, Any]:
    """Return the step's parameters with defaults filled in for omitted optional ones.

    Args:
        step: the plan step.
        spec: its capability spec.

    Returns:
        Parameter dict in registry order, followed by any unknown parameters
        (kept so G03 can still report them).
    """
    params: dict[str, Any] = {}
    for p in spec.params:
        if p.name in step.params:
            params[p.name] = step.params[p.name]
        elif not p.required:
            params[p.name] = p.default
    for name, value in step.params.items():
        if name not in params:
            params[name] = value
    return params


def fill_defaults(plan: Plan) -> Plan:
    """Return a copy of the plan with every optional parameter written out explicitly.

    The plan shown to the user and stored in the run record then fully describes
    the computation: there are no hidden defaults.

    Args:
        plan: the plan.

    Returns:
        A new Plan (the input is not modified). Steps with unknown capabilities
        are left as they are.
    """
    filled = plan.model_copy(deep=True)
    for step in filled.steps:
        spec = REGISTRY.get(step.capability)
        if spec is not None:
            step.params = effective_params(step, spec)
    return filled


def _has_type(value: Any, expected: str) -> bool:
    """Strict type check of a parameter value: "int", "float", "str" or "bool"."""
    if expected == "bool":
        return isinstance(value, bool)
    # bool is a subclass of int in Python (True == 1); reject it explicitly so
    # `"window": true` cannot silently mean window=1.
    if isinstance(value, bool):
        return False
    if expected == "int":
        return isinstance(value, int)
    if expected == "float":
        return isinstance(value, (int, float))  # 2 is an acceptable value for a float param
    return isinstance(value, str)


def param_problem(spec_param: ParamSpec, value: Any) -> str | None:
    """Describe what is wrong with one parameter value, or return None if it is valid.

    Args:
        spec_param: the parameter's spec.
        value: the value from the plan.

    Returns:
        An error message, or None.
    """
    name = spec_param.name
    if value is None:
        return f"'{name}' must not be null."
    if not _has_type(value, spec_param.type):
        return f"'{name}' must be {spec_param.type}, got {value!r} ({type(value).__name__})."
    if spec_param.choices is not None and value not in spec_param.choices:
        return f"'{name}'={value!r} is not one of {list(spec_param.choices)}."
    # Written as "not (min <= v <= max)" so NaN, which fails every comparison, is rejected.
    if spec_param.min is not None and not (spec_param.min <= value <= spec_param.max):
        return f"'{name}'={value!r} is outside [{spec_param.min:g}, {spec_param.max:g}]."
    return None


def _allowed_values(spec_param: ParamSpec) -> str:
    """Human-readable description of a parameter's allowed values."""
    if spec_param.choices is not None:
        return f"one of {list(spec_param.choices)}"
    if spec_param.min is not None:
        return f"{spec_param.type} in [{spec_param.min:g}, {spec_param.max:g}]"
    return spec_param.type


# ---------------------------------------------------------------------------
# Plan-level guardrails
# ---------------------------------------------------------------------------


def check_schema(raw: Any) -> tuple[Plan | None, list[Issue]]:
    """G01 INVALID_PLAN (schema): parse raw planner output into a Plan.

    Without it, malformed or partial LLM output would have to be guessed at, and
    fields the schema does not define (e.g. "code") could slip through unnoticed.

    Args:
        raw: the planner's output (normally a dict).

    Returns:
        (plan, []) if it parses, else (None, list of G01 issues, one per error).
    """
    try:
        return Plan.model_validate(raw), []
    except ValidationError as exc:
        issues = []
        for error in exc.errors():
            location = ".".join(str(part) for part in error["loc"]) or "(plan)"
            issues.append(
                make_issue(
                    "G01",
                    None,
                    f"Plan does not match the schema at {location}: {error['msg']}.",
                    "Regenerate the plan; the planner must return the plan schema exactly.",
                    key=location,
                )
            )
        return None, issues


def planner_failure_issue(error: str) -> Issue:
    """G01 INVALID_PLAN raised when the planner produced no usable plan at all.

    Without it, an API error or a missing tool call would crash the app instead
    of being shown and recorded like any other blocked plan.

    Args:
        error: the planner's error message.

    Returns:
        A G01 BLOCK issue.
    """
    return make_issue(
        "G01",
        None,
        f"The planner did not return a valid plan: {error}",
        "Retry, switch LLM mode (mock works offline), or rephrase the request.",
        key="planner",
    )


def check_planner_refused(plan: Plan) -> list[Issue]:
    """L01 PLANNER_REFUSED: the planner declared it cannot fulfil the request.

    The decision is the planner's (the LLM, or the mock planner for a request it
    has no canned plan for); this function turns it into a BLOCK, so a refused
    plan can never run, not even when it reaches the executor or replay. Without
    an explicit refusal path, the LLM is pushed to invent capabilities (or misuse
    existing ones) to satisfy every request.
    """
    if not plan.cannot_fulfill:
        return []
    return [
        make_issue(
            "L01",
            None,
            f"The planner declined the request: {plan.refusal_reason or 'no reason given'}",
            "Rephrase using the approved capabilities (see the Capability registry page).",
        )
    ]


def check_request_length(request: str, max_request_chars: int) -> list[Issue]:
    """G05 BUDGET_EXCEEDED (request size): refuse overly long requests.

    The engine calls it before planning, so an oversized request costs no LLM
    call, and validate_plan calls it again on every plan. Without it, a huge
    pasted request could blow up prompt size and cost, and gives more room for
    instructions hidden inside it.
    """
    if len(request) <= max_request_chars:
        return []
    return [
        make_issue(
            "G05",
            None,
            f"The request has {len(request)} characters; the limit is {max_request_chars}.",
            "Shorten the request.",
            key="request",
        )
    ]


def check_budget(plan: Plan, max_steps: int) -> list[Issue]:
    """G05 BUDGET_EXCEEDED (plan size): too many steps in one plan.

    Without it, a runaway plan could consume unbounded compute and become too
    long for a human to review before approving it. Every dataset is loaded by
    a step, so the step limit also bounds how much data a plan reads.
    """
    if len(plan.steps) <= max_steps:
        return []
    return [
        make_issue(
            "G05",
            None,
            f"The plan has {len(plan.steps)} steps; the limit is {max_steps}.",
            "Split the analysis into smaller requests.",
            key="steps",
        )
    ]


def check_unknown_datasets(plan: Plan, catalog: Catalog) -> list[Issue]:
    """G06 UNKNOWN_DATASET: every dataset id must exist in the catalog.

    Without it, the LLM could invent a dataset id or reach for data outside the
    approved, hash-verified catalog.
    """
    suggestion = f"Available datasets: {', '.join(catalog.ids())}."
    issues = [
        make_issue("G06", None, f"Dataset '{d}' is not in the catalog.", suggestion, key=d)
        for d in plan.dataset_ids
        if catalog.get(d) is None
    ]
    for step in plan.steps:
        dataset_id = step.params.get("dataset_id")
        is_load = step.capability == "load_series" and isinstance(dataset_id, str)
        if is_load and catalog.get(dataset_id) is None:
            issues.append(
                make_issue(
                    "G06", step.id, f"Dataset '{dataset_id}' is not in the catalog.", suggestion
                )
            )
    return issues


def check_not_empty(plan: Plan) -> list[Issue]:
    """G01 INVALID_PLAN (empty plan): a plan that is not a refusal must have steps.

    Without it, an empty plan would "succeed" with no results, which looks like
    an answer but is not one.
    """
    if plan.steps:
        return []
    return [
        make_issue(
            "G01",
            None,
            "The plan has no steps and is not marked as a refusal.",
            "Regenerate the plan.",
            key="empty",
        )
    ]


def check_duplicate_step_ids(plan: Plan) -> list[Issue]:
    """G01 INVALID_PLAN (duplicate ids): step ids must be unique.

    Without it, a reference like "s2" would be ambiguous and the wrong output
    could be fed into a later step.
    """
    seen: set[str] = set()
    issues = []
    for step in plan.steps:
        if step.id in seen:
            issues.append(
                make_issue(
                    "G01",
                    step.id,
                    f"Step id '{step.id}' is used more than once.",
                    "Number steps s1, s2, s3, ...",
                    key="duplicate",
                )
            )
        seen.add(step.id)
    return issues


# ---------------------------------------------------------------------------
# Step-level guardrails
# ---------------------------------------------------------------------------


def check_unknown_capability(step: Step) -> list[Issue]:
    """G02 UNKNOWN_CAPABILITY: the step names something that is not in the registry.

    Without it, the LLM could request an analysis nobody wrote or tested. (The
    executor could not run it anyway, since unregistered names have no function;
    this check reports it clearly before anything starts.)
    """
    if step.capability in REGISTRY:
        return []
    return [
        make_issue(
            "G02",
            step.id,
            f"'{step.capability}' is not a registered capability.",
            f"Use only: {', '.join(REGISTRY)}. New capabilities are added by developers, "
            "with tests, not by the planner.",
        )
    ]


def check_params(step: Step, spec: CapabilitySpec) -> list[Issue]:
    """G03 INVALID_PARAM: unknown, missing, wrongly typed or out-of-range parameters.

    Without it, typos, wrong types or extreme values (window=1, quantile=1.5)
    would crash mid-run or, worse, silently produce nonsense.
    """
    issues = []
    for name in step.params:
        if spec.param(name) is None:
            allowed = [p.name for p in spec.params] or "none"
            issues.append(
                make_issue(
                    "G03",
                    step.id,
                    f"Unknown parameter '{name}' for {spec.name}.",
                    f"Allowed parameters: {allowed}.",
                    key=name,
                )
            )
    for spec_param in spec.params:
        if spec_param.name not in step.params:
            if spec_param.required:
                issues.append(
                    make_issue(
                        "G03",
                        step.id,
                        f"Missing required parameter '{spec_param.name}' for {spec.name}.",
                        f"Provide '{spec_param.name}': {_allowed_values(spec_param)}.",
                        key=spec_param.name,
                    )
                )
            continue
        problem = param_problem(spec_param, step.params[spec_param.name])
        if problem:
            issues.append(
                make_issue(
                    "G03",
                    step.id,
                    problem,
                    f"Use {_allowed_values(spec_param)}.",
                    key=spec_param.name,
                )
            )
    return issues


def check_references(
    step: Step, spec: CapabilitySpec, earlier_ids: list[str], all_ids: list[str]
) -> list[Issue]:
    """G01 INVALID_PLAN (step references): inputs must point to EARLIER steps, all given.

    Without it, a step could read an output that does not exist yet (or ever),
    leaving the order of computation undefined.
    """
    issues = []
    for spec_input in spec.inputs:
        if spec_input.name not in step.inputs:
            issues.append(
                make_issue(
                    "G01",
                    step.id,
                    f"{spec.name} needs input '{spec_input.name}', which is not given.",
                    f"Set inputs['{spec_input.name}'] to the id of an earlier step.",
                    key=spec_input.name,
                )
            )
    for name, ref in step.inputs.items():
        if spec.input(name) is None:
            message = f"{spec.name} has no input called '{name}'."
            suggestion = f"Inputs of {spec.name}: {[i.name for i in spec.inputs] or 'none'}."
        elif ref in earlier_ids:
            continue
        elif ref in all_ids:
            message = f"Input '{name}' refers to step '{ref}', which is not an earlier step."
            suggestion = "Steps may only use outputs of steps that come before them."
        else:
            message = f"Input '{name}' refers to unknown step '{ref}'."
            suggestion = f"Earlier steps: {earlier_ids or 'none'}."
        issues.append(make_issue("G01", step.id, message, suggestion, key=name))
    return issues


def _type_hint(spec: CapabilitySpec, kind: SeriesKind | None) -> str:
    """A short, specific suggestion for common type mismatches (used by G04)."""
    if kind in (SeriesKind.RETURNS, SeriesKind.CHANGES) and spec.name in ("resample", "difference"):
        return "Resample or difference the LEVEL series first, then compute returns or changes."
    if spec.name == "rolling_volatility":
        return "Compute returns (prices) or differences (rates) first, then the volatility."
    return "Check the capability's allowed input kinds on the Capability registry page."


def check_input_types(
    step: Step, spec: CapabilitySpec, input_metas: dict[str, StaticMeta | None]
) -> list[Issue]:
    """G04 TYPE_MISMATCH: each input's data type and kind must be allowed by the capability.

    Without it, statistically meaningless chains would run without complaint,
    e.g. the volatility of a price level, or True/False flags used as a series.
    The outcome of hac_mean_difference gets its own message: the test compares
    means, and levels such as prices, index levels, rates or volatility trend
    and wander instead of fluctuating around a stable mean (they are
    non-stationary). The difference would then mostly reflect WHEN the flagged
    periods happened, and the p-value would be invalid.
    """
    issues = []
    for spec_input in spec.inputs:
        meta = input_metas.get(spec_input.name)
        if meta is None:
            continue  # missing or broken reference: already reported by G01/G02
        is_outcome = OUTCOME_INPUTS.get(spec.name) == spec_input.name
        if meta.kind == SeriesKind.FORWARD_RETURNS and not is_outcome:
            continue  # reported by the more specific G11 FORWARD_DATA_AS_SIGNAL
        if meta.data_type not in spec_input.data_types:
            expected = [t.value for t in spec_input.data_types]
            issues.append(
                make_issue(
                    "G04",
                    step.id,
                    f"Input '{spec_input.name}' of {spec.name} must be {expected}, "
                    f"got {meta.data_type.value}.",
                    _type_hint(spec, meta.kind),
                    key=spec_input.name,
                )
            )
        elif spec_input.kinds is not None and meta.kind not in spec_input.kinds:
            if spec.name == "compute_returns" and meta.kind == SeriesKind.RATE_LEVEL:
                continue  # reported by the more specific G07 RETURNS_ON_RATES
            expected = [k.value for k in spec_input.kinds]
            message = (
                f"Input '{spec_input.name}' of {spec.name} must be one of {expected}, "
                f"got kind '{meta.kind}'."
            )
            suggestion = _type_hint(spec, meta.kind)
            if spec.name == "hac_mean_difference" and is_outcome:
                message = (
                    f"The outcome of hac_mean_difference is a series of kind '{meta.kind}'; "
                    "it must be returns or forward returns. Levels such as prices, index "
                    "levels, rates or volatility are non-stationary, which makes the test "
                    "invalid."
                )
                suggestion = (
                    "Compute returns (compute_returns) or forward returns (forward_return) "
                    "of the price level, and test those."
                )
            issues.append(make_issue("G04", step.id, message, suggestion, key=spec_input.name))
    return issues


def check_returns_on_rates(
    step: Step, spec: CapabilitySpec, input_metas: dict[str, StaticMeta | None]
) -> list[Issue]:
    """G07 RETURNS_ON_RATES: compute_returns applied to an interest rate or yield.

    Without it, the system would happily compute "returns" of a yield: a move
    from 1% to 2% would be reported as +100%, which is meaningless. Rates must
    be differenced (changes in percentage points).
    """
    meta = input_metas.get("series")
    if spec.name != "compute_returns" or meta is None or meta.kind != SeriesKind.RATE_LEVEL:
        return []
    return [
        make_issue(
            "G07",
            step.id,
            "compute_returns on a rate/yield series: percentage 'returns' of a yield are "
            "meaningless.",
            "Use difference(series, periods=k) to get changes in percentage points.",
        )
    ]


def check_frequency_mismatch(
    step: Step, spec: CapabilitySpec, input_metas: dict[str, StaticMeta | None]
) -> list[Issue]:
    """G08 FREQUENCY_MISMATCH: two-input steps whose inputs have different frequencies.

    Without it, combining daily with weekly data silently keeps only the dates
    they share (Fridays), mixing two frequencies in one result.
    """
    if len(spec.inputs) < 2:
        return []
    # Inputs whose frequency is unknown are already reported elsewhere (G01/G03).
    metas = {name: m for name, m in input_metas.items() if m is not None and m.frequency}
    if len({meta.frequency for meta in metas.values()}) <= 1:
        return []
    described = ", ".join(
        f"'{name}' is {FREQUENCY_NAMES[meta.frequency]}" for name, meta in metas.items()
    )
    # Name the fix concretely: which input is too frequent, where it comes from,
    # and the frequency it should be resampled to (the least frequent one).
    target = max((meta.frequency for meta in metas.values()), key=FREQUENCY_RANK.get)
    finer = [name for name, meta in metas.items() if meta.frequency != target]
    sources = sorted({d for name in finer for d in metas[name].dataset_ids})
    suggestion = (
        f"Resample {', '.join(sources) or 'the source series'} to {FREQUENCY_NAMES[target]} "
        f"right after load_series (resample(freq='{target}')), before computing the "
        f"{', '.join(repr(name) for name in finer)} input."
    )
    message = f"{spec.name} inputs have different frequencies: {described}."
    return [make_issue("G08", step.id, message, suggestion)]


def check_look_ahead(
    step: Step,
    spec: CapabilitySpec,
    params: dict[str, Any],
    input_metas: dict[str, StaticMeta | None],
    tested_in: list[str],
) -> list[Issue]:
    """G09 LOOK_AHEAD: flag_regime with a full-sample threshold (needs approval).

    Without it, a threshold computed from the whole sample (including the future)
    would be used silently; flags then use information nobody had at the time,
    and any backtest looks better than reality. When the flags feed a significance
    test (`tested_in`), the message says that the test is invalidated too.
    """
    if spec.name != "flag_regime" or params.get("method") != "full_sample":
        return []
    meta = input_metas.get("series")
    period = PERIOD_NAMES.get(meta.frequency, "period") if meta else "period"
    quantile = params.get("quantile")
    message = (
        f"A full-sample threshold judges past {period}s using information from the future: "
        f"the threshold (the {quantile:.0%} quantile) is computed from every {period}, "
        "including those that "
        f"came later, so a {period} flagged here could not have been flagged at the time "
        "(look-ahead bias). Results will look better than anything achievable in real time."
    )
    if tested_in:
        message += (
            f" It also invalidates the significance test in {', '.join(tested_in)}: the "
            "test assumes the flag was known at t, and this flag would not have been."
        )
    suggestion = (
        "Use method='expanding': each threshold then uses only the data before that "
        f"{period}. Keep full_sample only for purely descriptive labelling."
    )
    return [make_issue("G09", step.id, message, suggestion)]


def check_forward_data_as_signal(
    step: Step, spec: CapabilitySpec, input_metas: dict[str, StaticMeta | None]
) -> list[Issue]:
    """G11 FORWARD_DATA_AS_SIGNAL: forward returns used anywhere but as an outcome.

    Without it, a forward return (the return from t to t+1, stamped at t) could
    feed a volatility, a flag or another transformation. Anything built from it
    uses information nobody had at t, so a "signal" would look predictive by
    construction. Forward returns may only be the outcome input of compare_groups
    or hac_mean_difference.
    """
    issues = []
    for name, meta in input_metas.items():
        if meta is None or meta.kind != SeriesKind.FORWARD_RETURNS:
            continue
        if OUTCOME_INPUTS.get(spec.name) == name:
            continue
        issues.append(
            make_issue(
                "G11",
                step.id,
                f"Input '{name}' of {spec.name} is a forward return: forward returns contain "
                "future information and can only be used as an outcome, never to build a "
                "signal.",
                "Build signals (volatility, flags) from returns known at t (compute_returns); "
                "use forward_return only as the values of compare_groups or the outcome of "
                "hac_mean_difference.",
                key=name,
            )
        )
    return issues


def check_material_assumptions(
    step: Step, spec: CapabilitySpec, params: dict[str, Any]
) -> list[Issue]:
    """G10 MATERIAL_ASSUMPTION: a capability assumption that a human must approve.

    Without it, material modelling choices (e.g. simple vs log returns) would be
    made by the LLM and could change conclusions without anybody noticing.
    """
    issues = []
    for assumption in spec.assumptions:
        if not assumption.requires_approval:
            continue
        chosen = f" Chosen: {assumption.param}={params.get(assumption.param)!r}."
        issues.append(
            make_issue(
                "G10",
                step.id,
                assumption.text + (chosen if assumption.param else ""),
                "Approve if this choice is intended, or edit the parameter.",
                key=assumption.id,
            )
        )
    return issues


# ---------------------------------------------------------------------------
# User edits: structure lock (G03)
# ---------------------------------------------------------------------------


def check_edits(plan: Plan, edits: Any) -> list[Issue]:
    """G03 INVALID_PARAM (edit rejected): users may change parameter VALUES only.

    Without it, an edit could add, remove or reorder steps, change a step's
    capability or input references, or smuggle in a field that becomes code,
    and the plan the human approved would no longer be the plan the LLM and the
    validator saw. Values themselves are not judged here: they are applied and
    then checked by the validator (G03 for out-of-range values, G06 for datasets...).

    Args:
        plan: the current plan.
        edits: the requested edits; must be {step_id: {param_name: value}}.

    Returns:
        One BLOCK issue per rejected edit (empty if every edit is a parameter value).
    """

    def rejected(message: str, key: str) -> Issue:
        suggestion = "Only parameter values can be edited; regenerate the plan to change its steps."
        return make_issue("G03", None, f"Edit rejected: {message}", suggestion, key=f"edit:{key}")

    if not isinstance(edits, dict):
        return [rejected("edits must be {step_id: {parameter: value}}.", "shape")]
    steps = {step.id: step for step in plan.steps}
    issues = []
    for step_id, params in edits.items():
        step = steps.get(step_id)
        if step is None:
            issues.append(rejected(f"step '{step_id}' does not exist; steps cannot be added.",
                                   str(step_id)))  # fmt: skip
            continue
        if not isinstance(params, dict):
            issues.append(rejected(f"edits for {step_id} must be {{parameter: value}}.", step_id))
            continue
        spec = REGISTRY.get(step.capability)
        for name in params:
            if spec is None or spec.param(name) is None:
                # Covers "capability", "inputs", "id" and any invented field.
                issues.append(rejected(f"'{name}' is not a parameter of {step_id} "
                                       f"({step.capability}).", f"{step_id}.{name}"))  # fmt: skip
    return issues


# ---------------------------------------------------------------------------
# Dry run: predict output metadata without data
# ---------------------------------------------------------------------------


def infer_output_meta(
    spec: CapabilitySpec,
    params: dict[str, Any],
    input_metas: dict[str, StaticMeta | None],
    catalog: Catalog,
) -> StaticMeta | None:
    """Predict a step's output metadata from its inputs' metadata, params and the registry.

    Mirrors what the capability functions set at run time (a test checks they agree).

    Args:
        spec: the capability.
        params: effective parameters (defaults filled).
        input_metas: input name -> predicted metadata of the referenced step.
        catalog: the dataset catalog (for load_series).

    Returns:
        The predicted StaticMeta, or None when it cannot be predicted because an
        input is missing or invalid (that problem is reported elsewhere).
    """
    if spec.output_kind == FROM_CATALOG:
        dataset_id = params.get("dataset_id")
        entry = catalog.get(dataset_id) if isinstance(dataset_id, str) else None
        if entry is None:
            return None
        return StaticMeta(
            data_type=spec.output_type,
            kind=entry.kind,
            frequency=entry.frequency,
            dataset_ids=[entry.dataset_id],
        )

    metas = [input_metas.get(spec_input.name) for spec_input in spec.inputs]
    if any(meta is None for meta in metas):
        return None
    first = metas[0]  # metadata is inherited from the first input, as at run time
    datasets = list(dict.fromkeys(d for meta in metas for d in meta.dataset_ids))
    predicted = StaticMeta(
        data_type=spec.output_type,
        kind=first.kind if spec.output_kind == SAME_AS_INPUT else spec.output_kind,
        frequency=first.frequency,
        return_method=first.return_method,
        periods=first.periods,
        horizon=first.horizon,
        dataset_ids=datasets,
    )
    for meta_field, param_name in spec.meta_from_params.items():
        value = params.get(param_name)
        if meta_field == "frequency":
            value = Frequency(value) if value in ("D", "W", "M") else None
        setattr(predicted, meta_field, value)
    if spec.output_kind == SeriesKind.FORWARD_RETURNS:
        predicted.horizon = FORWARD_HORIZON  # no parameter: always one period ahead
    return predicted


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def validate_plan(
    plan: Plan,
    catalog: Catalog,
    *,
    max_steps: int = 15,
    max_request_chars: int = 2000,
) -> ValidationReport:
    """Run every validator guardrail (G codes, and the L01 refusal block) on a plan.

    Args:
        plan: the plan to check (defaults may or may not be filled in).
        catalog: the dataset catalog.
        max_steps: budget for G05.
        max_request_chars: budget for G05.

    Returns:
        A ValidationReport with all issues (not just the first) and the dry-run metadata.
    """
    issues = check_planner_refused(plan)
    if plan.cannot_fulfill:
        return ValidationReport(issues=issues)

    issues += check_not_empty(plan)
    issues += check_request_length(plan.request, max_request_chars)
    issues += check_budget(plan, max_steps)
    issues += check_unknown_datasets(plan, catalog)
    issues += check_duplicate_step_ids(plan)

    all_ids = [step.id for step in plan.steps]
    # Flag step id -> the significance tests that use its flags (for G09's message).
    tested_in: dict[str, list[str]] = {}
    for step in plan.steps:
        if step.capability == "hac_mean_difference" and "flags" in step.inputs:
            tested_in.setdefault(step.inputs["flags"], []).append(step.id)
    metas: dict[str, StaticMeta | None] = {}
    for step in plan.steps:
        earlier_ids = list(metas)
        spec = REGISTRY.get(step.capability)
        if spec is None:
            issues += check_unknown_capability(step)
            metas[step.id] = None
            continue

        params = effective_params(step, spec)
        input_metas = {name: metas[ref] for name, ref in step.inputs.items() if ref in earlier_ids}
        issues += check_params(step, spec)
        issues += check_references(step, spec, earlier_ids, all_ids)
        issues += check_returns_on_rates(step, spec, input_metas)
        issues += check_forward_data_as_signal(step, spec, input_metas)
        issues += check_input_types(step, spec, input_metas)
        issues += check_frequency_mismatch(step, spec, input_metas)
        issues += check_look_ahead(step, spec, params, input_metas, tested_in.get(step.id, []))
        issues += check_material_assumptions(step, spec, params)
        metas[step.id] = infer_output_meta(spec, params, input_metas, catalog)

    dry_run = {step_id: meta.to_dict() for step_id, meta in metas.items() if meta is not None}
    return ValidationReport(issues=issues, dry_run=dry_run)
