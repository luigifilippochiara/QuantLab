"""Planner evaluation: does the planner pick the right plan, and refuse only when it should?

Role in the architecture: this measures PLANNER accuracy only. It runs the planner
(mock or live) on a set of requests, validates each plan, and scores it. It does not
execute anything: whether the computations themselves are right is measured
separately, by the unit tests.

Two sets share one schema (EvalCase), one scorer and one report:
- the golden set (planner_eval/golden_requests.yaml): known requests and the plan each
  should get, scored against the expected capability sequence, key parameters and
  (where given) the expected guardrail outcome;
- the adversarial set (planner_eval/adversarial_requests.yaml): the boundary, where
  vague, out-of-scope, unavailable-data, advice, injection, off-topic and mixed
  requests must be refused (P01), and valid but loosely worded ones must be planned.
Every case has a category, and may be held out of prompt work: the report gives the
accuracy per category, development and held-out cases apart.

A case expects one of three outcomes:
- plan: a valid plan (no BLOCK); when the case gives the plan (`capabilities`), also
  that exact capability sequence and, where given, its guardrail outcome;
- refuse: the planner itself declines (P01, or R01 before it). A plan the validator
  blocks is safe, but not correct: the planner did not refuse;
- refuse_or_block: anything but a valid plan (a refusal, a block, or no plan at all).

Metrics: valid-plan rate (cases expecting a plan); exact sequence match and
capability-set Jaccard (|expected AND planned| / |expected OR planned|, averaged), over
the cases that give the plan; key-parameter accuracy (share of expected parameter
values found in the plan); guardrail-outcome match; correct refusal rate (cases
expecting a refusal or a block); overall correct. With them, the counts of false plans,
false refusals, blocks and planner failures, and the tokens and cost of the run.

Each case that gives a plan or an outcome is also scored item by item (every step in
order, every key parameter, each action of the guardrail outcome), so a mismatch shows
exactly what the planner chose differently. The CLI (`quantlab.cli eval`) and
planner_eval/run_eval.py both call `run_evaluation`.
"""

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from quantlab.config import LLM_PRICES_PER_MILLION_TOKENS, PROJECT_ROOT, Settings
from quantlab.data import Catalog, load_catalog
from quantlab.guardrails import Action, Issue, check_request_length
from quantlab.mock_scenarios import SCENARIOS_BY_ID
from quantlab.monitoring import token_cost
from quantlab.plan import Plan
from quantlab.planner import PROMPT_VERSION, PlanResult, generate_plan
from quantlab.registry import REGISTRY
from quantlab.validator import effective_params, fill_defaults, validate_plan

GOLDEN_PATH = PROJECT_ROOT / "planner_eval" / "golden_requests.yaml"
ADVERSARIAL_PATH = PROJECT_ROOT / "planner_eval" / "adversarial_requests.yaml"
REPORTS_DIR = PROJECT_ROOT / "planner_eval" / "reports"

# Each set's categories, in report order.
GOLDEN_CATEGORIES = (
    "example",
    "variation",
    "paraphrase",
    "capability",
    "parameter",
    "trap",
    "dataset",
    "refusal",
)
ADVERSARIAL_CATEGORIES = (
    "vague",
    "out_of_scope_method",
    "data_unavailable",
    "advice",
    "injection",
    "off_topic",
    "mixed",
    "valid_loose",
)

# What a case may expect (see the module docstring).
EXPECTATIONS = ("plan", "refuse", "refuse_or_block")
# The cases a run scores: all of them, the development cases only, or the held-out ones.
HOLDOUT_CHOICES = ("include", "exclude", "only")

# The keys of a case's `guardrails` entry (one per action), and the item names they
# are reported under.
GUARDRAIL_ITEMS = {
    "approve": (Action.APPROVE, "approvals"),
    "warn": (Action.WARN, "warnings"),
    "block": (Action.BLOCK, "blocks"),
}


@dataclass(frozen=True)
class EvalSet:
    """One evaluation set.

    Fields:
        name: "golden" or "adversarial" (the CLI's --set, and part of the report's name).
        path: its YAML file.
        categories: its categories, in report order.
    """

    name: str
    path: Path
    categories: tuple[str, ...]


GOLDEN = EvalSet("golden", GOLDEN_PATH, GOLDEN_CATEGORIES)
ADVERSARIAL = EvalSet("adversarial", ADVERSARIAL_PATH, ADVERSARIAL_CATEGORIES)
EVAL_SETS = {eval_set.name: eval_set for eval_set in (GOLDEN, ADVERSARIAL)}


@dataclass
class EvalCase:
    """One request of an evaluation set (the YAML files' headers describe the fields).

    Fields:
        id: short name, unique in its set.
        category: one of its set's categories.
        request: the request, as an analyst would type it.
        datasets: datasets selected in the app ([]: none, so every dataset is offered).
        expect: "plan", "refuse" or "refuse_or_block".
        holdout: kept out of prompt work, and scored apart (--holdout).
        note: why the expected outcome is the right one ("" when it is obvious).
        capabilities: the expected capability sequence, or None when the case does not
            give the plan (then only its outcome is scored).
        key_params: capability -> {param: expected value}.
        guardrails: the expected issue codes of each action, or None.
    """

    id: str
    category: str
    request: str
    datasets: list[str]
    expect: str
    holdout: bool = False
    note: str = ""
    capabilities: list[str] | None = None
    key_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    guardrails: dict[str, list[str]] | None = None


@dataclass
class ItemCheck:
    """One scored item of a case: what was expected and what the planner chose.

    Fields:
        item: what is checked, e.g. "step 6", "forward_return.method" or "approvals".
        expected: the expected value, as text.
        actual: the planned value, as text.
        match: whether they agree.
    """

    item: str
    expected: str
    actual: str
    match: bool


@dataclass
class CaseScore:
    """How the planner did on one case.

    Fields:
        case: the case.
        outcome: "refused" (P01, or R01 before the planner), "planned" (a valid plan),
            "blocked" (a plan the validator blocks) or "failed" (no usable output, V01).
        detail: one line on what happened: the refusal reason, the planned
            capabilities, the blocking codes, or the error.
        planned: capability sequence the planner produced ([] if none).
        issue_codes: validation issue codes (the planner's own, when there is no plan).
        valid: a valid plan: not a refusal, no BLOCK.
        refused_or_blocked: anything but a valid plan (a refusal, a block, no plan).
        sequence_match: capability sequence equals the expected one ([] if none is given).
        jaccard: capability-set overlap with the expected set.
        params_hit: expected parameter values found.
        params_total: expected parameter values checked.
        correct: the expected outcome was reached (see score_case).
        error: planner error, if any.
        guardrails_match: issues are exactly the expected ones (None if not specified).
        checks: the item-by-item comparison (cases that give a plan or an outcome).
        model: the model that answered (None in mock mode, or when R01 stopped the request).
        input_tokens: input tokens reported by the API (0 without a call, None if unknown).
        output_tokens: output tokens reported by the API.
    """

    case: EvalCase
    outcome: str
    detail: str
    planned: list[str]
    issue_codes: list[str]
    valid: bool
    refused_or_blocked: bool
    sequence_match: bool
    jaccard: float
    params_hit: int
    params_total: int
    correct: bool
    error: str | None
    guardrails_match: bool | None = None
    checks: list[ItemCheck] = field(default_factory=list)
    model: str | None = None
    input_tokens: int | None = 0
    output_tokens: int | None = 0


@dataclass
class EvalReport:
    """One run of an evaluation set.

    Fields:
        eval_set: the set's name.
        mode: LLM mode.
        model: the planner model ("mock" in mock mode).
        prompt_version: the planner prompt's version.
        created_at: UTC timestamp.
        holdout: which cases were scored: "include", "exclude" or "only".
        scores: one per case.
        metrics: the headline rates (see summarise).
        summary: the counts per outcome and category, tokens and cost (see count_outcomes).
    """

    eval_set: str
    mode: str
    model: str
    prompt_version: str
    created_at: str
    holdout: str
    scores: list[CaseScore]
    metrics: dict[str, float | None]
    summary: dict[str, Any]


def load_cases(eval_set: EvalSet = GOLDEN) -> list[EvalCase]:
    """Read an evaluation set, resolving `scenario:` references to their request text.

    Args:
        eval_set: the set (GOLDEN or ADVERSARIAL).

    Returns:
        The cases, in file order.

    Raises:
        ValueError: on an unknown category or expectation, a repeated id, or a
            `guardrails` key other than approve, warn or block.
    """
    cases = []
    for item in yaml.safe_load(eval_set.path.read_text(encoding="utf-8")):
        if "scenario" in item:
            scenario = SCENARIOS_BY_ID[item["scenario"]]
            request, datasets = scenario.request, list(scenario.dataset_ids)
        else:
            request, datasets = item["request"], list(item["datasets"])
        case = EvalCase(
            id=item["id"],
            category=item["category"],
            request=request,
            datasets=datasets,
            expect=item["expect"],
            holdout=bool(item["holdout"]),
            note=item.get("note", ""),
            capabilities=item.get("capabilities"),
            key_params=item.get("key_params", {}),
            guardrails=item.get("guardrails"),
        )
        if case.category not in eval_set.categories:
            raise ValueError(f"{case.id}: unknown category {case.category!r}")
        if case.expect not in EXPECTATIONS:
            raise ValueError(f"{case.id}: expect must be one of {list(EXPECTATIONS)}, "
                             f"not {case.expect!r}")  # fmt: skip
        unknown = sorted(set(case.guardrails or {}) - set(GUARDRAIL_ITEMS))
        if unknown:
            # A misspelt key would otherwise be silently ignored, and pass as "none".
            raise ValueError(f"{case.id}: unknown guardrails key(s) {unknown}; "
                             f"use {list(GUARDRAIL_ITEMS)}.")  # fmt: skip
        cases.append(case)
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{eval_set.name} case ids must be unique")
    return cases


def _same_value(expected: Any, actual: Any) -> bool:
    """Parameter equality: floats with a small tolerance, everything else exactly."""
    if isinstance(expected, float) or isinstance(actual, float):
        is_number = isinstance(actual, (int, float)) and not isinstance(actual, bool)
        return is_number and math.isclose(expected, actual, abs_tol=1e-9)
    return expected == actual


def sequence_checks(expected: list[str], planned: list[str]) -> list[ItemCheck]:
    """Compare the capability sequence position by position ("(none)" past either end)."""
    checks = []
    for position in range(max(len(expected), len(planned))):
        want = expected[position] if position < len(expected) else "(none)"
        got = planned[position] if position < len(planned) else "(none)"
        checks.append(ItemCheck(f"step {position + 1}", want, got, want == got))
    return checks


def param_checks(key_params: dict[str, dict[str, Any]], plan: Plan | None) -> list[ItemCheck]:
    """Compare each expected parameter value with the plan (defaults filled in).

    A match only if EVERY step using the capability has the expected value (e.g. if
    a plan resamples two series, both must be resampled weekly).
    """
    checks = []
    for capability, params in key_params.items():
        steps = [step for step in plan.steps if step.capability == capability] if plan else []
        for name, expected in params.items():
            values = [effective_params(s, REGISTRY[capability]).get(name) for s in steps]
            match = bool(steps) and all(_same_value(expected, value) for value in values)
            actual = ", ".join(str(value) for value in values) or f"(no {capability} step)"
            checks.append(ItemCheck(f"{capability}.{name}", str(expected), actual, match))
    return checks


def guardrail_checks(expected: dict[str, list[str]], issues: list[Issue]) -> list[ItemCheck]:
    """Compare the issue codes of each action with the expected ones (order ignored)."""
    checks = []
    for key, (action, item) in GUARDRAIL_ITEMS.items():
        want = sorted(expected.get(key, []))
        got = sorted(i.code for i in issues if i.action == action)
        checks.append(ItemCheck(item, ", ".join(want) or "none", ", ".join(got) or "none",
                                want == got))  # fmt: skip
    return checks


def score_case(
    case: EvalCase, settings: Settings, catalog: Catalog, mode: str, client: Any = None
) -> CaseScore:
    """Plan one case's request, validate the plan as the graph does, and score it.

    Correct means, by what the case expects: "plan", a valid plan and, when the case
    gives the plan, exactly its capability sequence and (where given) its guardrail
    outcome; "refuse", a refusal by the planner (or R01); "refuse_or_block", anything
    but a valid plan.

    Args:
        case: the case.
        settings: settings.
        catalog: dataset catalog.
        mode: "mock", "cached" or "live".
        client: optional OpenAI or Anthropic client (tests inject a fake one).

    Returns:
        The CaseScore.
    """
    # R01, as in the graph's request check: an oversized request never reaches the planner.
    too_long = check_request_length(case.request)
    if too_long:
        result = PlanResult(plan=None, issues=too_long, error=too_long[0].message)
    else:
        result = generate_plan(case.request, case.datasets, mode, settings, catalog, client)
    plan = fill_defaults(result.plan) if result.plan else None
    report = validate_plan(plan, catalog, max_steps=settings.max_steps) if plan else None

    planned = [step.capability for step in plan.steps] if plan else []
    codes = report.codes() if report else sorted({i.code for i in result.issues})
    if too_long:
        outcome, detail = "refused", too_long[0].message
    elif plan is None:
        outcome, detail = "failed", result.error or "no plan"
    elif plan.cannot_fulfill:
        outcome, detail = "refused", (plan.refusal_reason or "").replace("\n", " ")
    elif report.is_blocked:
        blocking = ", ".join(sorted({issue.code for issue in report.blocks}))
        outcome, detail = "blocked", f"blocked by {blocking}: {' > '.join(planned)}"
    else:
        outcome, detail = "planned", " > ".join(planned)
    valid = outcome == "planned"

    expected = case.capabilities or []
    expected_set, planned_set = set(expected), set(planned)
    union = expected_set | planned_set
    jaccard = len(expected_set & planned_set) / len(union) if union else 1.0

    params = param_checks(case.key_params, plan)
    guardrails_match = None
    checks: list[ItemCheck] = []
    gives_plan = case.expect == "plan" and case.capabilities is not None
    if gives_plan:
        checks = sequence_checks(expected, planned) + params
    if case.guardrails is not None:
        # A case expecting a refusal or a block that gives an outcome (an example
        # workflow) is reported item by item too, for information: it stays correct if
        # refused or blocked for another reason, e.g. a live planner refusing what the
        # canned plan lets the validator block.
        if case.expect != "plan":
            checks = sequence_checks(expected, planned)
        issues = report.issues if report else result.issues
        outcome_checks = guardrail_checks(case.guardrails, issues)
        guardrails_match = all(check.match for check in outcome_checks)
        checks += outcome_checks

    sequence_match = planned == expected
    if case.expect == "plan":
        correct = valid and (sequence_match or not gives_plan) and guardrails_match is not False
    elif case.expect == "refuse":
        correct = outcome == "refused"
    else:
        correct = not valid

    if too_long or mode == "mock":
        model, input_tokens, output_tokens = None, 0, 0
    else:
        # A plan served from the cache made no call, so it costs nothing.
        model = result.served_model
        input_tokens = 0 if result.from_cache else result.input_tokens
        output_tokens = 0 if result.from_cache else result.output_tokens
    return CaseScore(
        case=case,
        outcome=outcome,
        detail=detail,
        planned=planned,
        issue_codes=codes,
        valid=valid,
        refused_or_blocked=not valid,
        sequence_match=sequence_match,
        jaccard=round(jaccard, 4),
        params_hit=sum(check.match for check in params),
        params_total=len(params),
        correct=correct,
        error=result.error,
        guardrails_match=guardrails_match,
        checks=checks,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def summarise(scores: list[CaseScore]) -> dict[str, float | None]:
    """Aggregate the per-case scores into the headline rates (None when no case applies)."""
    expect_plan = [s for s in scores if s.case.expect == "plan"]
    give_plan = [s for s in expect_plan if s.case.capabilities is not None]
    expect_refusal = [s for s in scores if s.case.expect != "plan"]

    def rate(values: list[bool]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    params_total = sum(s.params_total for s in expect_plan)
    return {
        "valid_plan_rate": rate([s.valid for s in expect_plan]),
        "exact_sequence_match": rate([s.sequence_match for s in give_plan]),
        "mean_jaccard": (
            round(sum(s.jaccard for s in give_plan) / len(give_plan), 4) if give_plan else None
        ),
        "key_param_accuracy": (
            round(sum(s.params_hit for s in expect_plan) / params_total, 4)
            if params_total
            else None
        ),
        "guardrail_outcome_match": rate(
            [s.guardrails_match for s in expect_plan if s.guardrails_match is not None]
        ),
        "correct_refusal_block_rate": rate([s.correct for s in expect_refusal]),
        "overall_correct": rate([s.correct for s in scores]),
    }


def count_outcomes(scores: list[CaseScore], categories: tuple[str, ...]) -> dict[str, Any]:
    """Accuracy per category (all, development and holdout cases), errors and cost.

    Args:
        scores: the per-case scores.
        categories: the set's categories, in report order.

    Returns:
        A dict with: "categories" (category -> {"cases", "correct", "dev", "dev_correct",
        "holdout", "holdout_correct"}), "cases", "correct", "false_plans" (planned when
        it should refuse), "false_refusals" (refused when it should plan), "blocked",
        "failed", "input_tokens", "output_tokens" (None if any count is unknown) and
        "cost_usd" (None if not every call has a price).
    """
    by_category: dict[str, dict[str, int]] = {}
    for name in categories:
        in_category = [s for s in scores if s.case.category == name]
        dev = [s for s in in_category if not s.case.holdout]
        held = [s for s in in_category if s.case.holdout]
        by_category[name] = {
            "cases": len(in_category),
            "correct": sum(s.correct for s in in_category),
            "dev": len(dev),
            "dev_correct": sum(s.correct for s in dev),
            "holdout": len(held),
            "holdout_correct": sum(s.correct for s in held),
        }
    inputs = [s.input_tokens for s in scores]
    outputs = [s.output_tokens for s in scores]
    costs = [
        token_cost(s.model, s.input_tokens, s.output_tokens, LLM_PRICES_PER_MILLION_TOKENS)
        if s.model
        else 0.0
        for s in scores
    ]
    return {
        "categories": by_category,
        "cases": len(scores),
        "correct": sum(s.correct for s in scores),
        "false_plans": sum(s.case.expect != "plan" and s.outcome == "planned" for s in scores),
        "false_refusals": sum(s.case.expect == "plan" and s.outcome == "refused" for s in scores),
        "blocked": sum(s.outcome == "blocked" for s in scores),
        "failed": sum(s.outcome == "failed" for s in scores),
        "input_tokens": None if None in inputs else sum(inputs),
        "output_tokens": None if None in outputs else sum(outputs),
        "cost_usd": None if None in costs else sum(costs),
    }


def run_evaluation(
    mode: str,
    settings: Settings,
    eval_set: EvalSet = GOLDEN,
    case_ids: list[str] | None = None,
    holdout: str = "include",
) -> EvalReport:
    """Run the planner on an evaluation set and compute the metrics.

    Args:
        mode: "mock", "cached" or "live".
        settings: settings.
        eval_set: the set to run (GOLDEN or ADVERSARIAL).
        case_ids: evaluate only these cases (None = all of them).
        holdout: "include" (every case), "exclude" (development cases only, for prompt
            work) or "only" (the held-out cases).

    Returns:
        The EvalReport.

    Raises:
        ValueError: if a requested case id is not in the set.
    """
    catalog = load_catalog(settings.catalog_path)
    cases = load_cases(eval_set)
    if case_ids is not None:
        unknown = sorted(set(case_ids) - {case.id for case in cases})
        if unknown:
            raise ValueError(f"unknown {eval_set.name} case id(s): {', '.join(unknown)}")
        cases = [case for case in cases if case.id in case_ids]
    if holdout == "exclude":
        cases = [case for case in cases if not case.holdout]
    elif holdout == "only":
        cases = [case for case in cases if case.holdout]
    scores = [score_case(case, settings, catalog, mode) for case in cases]
    return EvalReport(
        eval_set=eval_set.name,
        mode=mode,
        model="mock" if mode == "mock" else settings.planner_model,
        prompt_version=PROMPT_VERSION,
        created_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        holdout=holdout,
        scores=scores,
        metrics=summarise(scores),
        summary=count_outcomes(scores, eval_set.categories),
    )


def _percent(value: float | None) -> str:
    """Format a rate as a percentage, or "n/a"."""
    return "n/a" if value is None else f"{value:.0%}"


def _share(correct: int, total: int) -> str:
    """ "3/4 (75%)", or "-" for no cases."""
    return f"{correct}/{total} ({correct / total:.0%})" if total else "-"


def write_report(report: EvalReport, reports_dir: Path = REPORTS_DIR) -> Path:
    """Write an evaluation run as a Markdown report named after its timestamp and set.

    Args:
        report: the evaluation run.
        reports_dir: output folder.

    Returns:
        Path of the written file.
    """
    m, s = report.metrics, report.summary
    tokens = (
        f"{s['input_tokens']:,} in, {s['output_tokens']:,} out"
        if s["input_tokens"] is not None and s["output_tokens"] is not None
        else "unknown"
    )
    cost = "unknown" if s["cost_usd"] is None else f"${s['cost_usd']:.4f}"
    held = sum(score.case.holdout for score in report.scores)
    scope = "" if report.holdout == "include" else f", --holdout {report.holdout}"
    lines = [
        f"# Planner evaluation: {report.eval_set} set ({report.mode})",
        "",
        f"- Created: {report.created_at}",
        f"- Planner: {report.model}, prompt {report.prompt_version}",
        f"- Cases: {s['cases']} ({held} held out{scope}); "
        f"correct: {_share(s['correct'], s['cases'])}",
        f"- False plans (planned when it should refuse): {s['false_plans']}",
        f"- False refusals (refused when it should plan): {s['false_refusals']}",
        f"- Blocked by the validator (not refused by the planner): {s['blocked']}",
        f"- Planner failures: {s['failed']}",
        f"- Tokens: {tokens}; cost: {cost}",
        "",
    ]
    if report.mode == "mock":
        lines += [
            "> Mock mode tests the evaluation harness, not an LLM: the mock planner only",
            "> knows the five example workflows, and refuses everything else. Run with",
            "> `--mode live` to measure the real planner.",
            "",
        ]
    lines += [
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Valid-plan rate (cases expecting a plan) | {_percent(m['valid_plan_rate'])} |",
        f"| Exact capability-sequence match | {_percent(m['exact_sequence_match'])} |",
        f"| Mean capability-set Jaccard | {m['mean_jaccard']} |",
        f"| Key-parameter accuracy | {_percent(m['key_param_accuracy'])} |",
        f"| Guardrail-outcome match | {_percent(m['guardrail_outcome_match'])} |",
        f"| Correct refusal rate (cases expecting a refusal or a block) | "
        f"{_percent(m['correct_refusal_block_rate'])} |",
        f"| Overall correct | {_percent(m['overall_correct'])} |",
        "",
        "## Accuracy per category",
        "",
        "| Category | All | Development | Holdout |",
        "|---|---|---|---|",
    ]
    for name, c in s["categories"].items():
        if c["cases"]:
            lines.append(
                f"| {name} | {_share(c['correct'], c['cases'])} | "
                f"{_share(c['dev_correct'], c['dev'])} | "
                f"{_share(c['holdout_correct'], c['holdout'])} |"
            )
    lines += [
        "",
        "## Cases",
        "",
        "| Case | Category | Holdout | Expect | Outcome | Correct | Sequence | Params | "
        "Issues | Planned |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for score in report.scores:
        case = score.case
        sequence = "-" if case.capabilities is None else ("yes" if score.sequence_match else "no")
        params = f"{score.params_hit}/{score.params_total}" if score.params_total else "-"
        lines.append(
            f"| {case.id} | {case.category} | {'yes' if case.holdout else ''} | {case.expect} | "
            f"{score.outcome} | {'yes' if score.correct else 'NO'} | {sequence} | {params} | "
            f"{', '.join(score.issue_codes) or '-'} | {' > '.join(score.planned) or '(none)'} |"
        )
    wrong = [score for score in report.scores if not score.correct]
    if wrong:
        lines += ["", "## Wrong answers", ""]
        for score in wrong:
            why = f" Expected because: {score.case.note}" if score.case.note else ""
            detail = score.detail.rstrip(".")  # a refusal reason may end with its own
            lines.append(f"- **{score.case.id}** (expected {score.case.expect}, "
                         f"{score.outcome}): {detail}.{why}")  # fmt: skip
    lines += ["", "## Item checks", ""]
    for score in report.scores:
        if not score.checks:
            continue
        lines += [f"### {score.case.id}", "", "| Item | Expected | Planned | Match |"]
        lines.append("|---|---|---|---|")
        lines += [
            f"| {c.item} | {c.expected} | {c.actual} | {'yes' if c.match else 'NO'} |"
            for c in score.checks
        ]
        lines.append("")
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path = reports_dir / f"{stamp}-{report.eval_set}-{report.mode}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
