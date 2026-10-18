"""Planner evaluation: does the planner pick the right plan for known requests?

Role in the architecture: this measures PLANNER accuracy only. It runs the
planner (mock or live) on the golden requests in eval/golden_requests.yaml,
validates each plan, and scores it against the expected capability sequence,
key parameters and (where given) the expected guardrail outcome. It does not
execute anything: whether the computations themselves are right is measured
separately, by the unit tests.

Metrics (normal cases = expect "plan"; adversarial = expect "refuse_or_block"):
- valid-plan rate: normal cases whose plan has no BLOCK and is not a refusal;
- exact sequence match: capability list equals the expected list, in order;
- capability-set Jaccard: |expected AND planned| / |expected OR planned|, averaged;
- key-param accuracy: share of expected parameter values found in the plan;
- guardrail-outcome match: cases whose validation issues are exactly the expected ones;
- correct refusal/block rate: adversarial cases that were refused or blocked.

Each normal case is also scored item by item (every step in order, every key
parameter, each action of the guardrail outcome), so a mismatch shows exactly
what the planner chose differently. The CLI (`quantlab.cli eval`) and
eval/run_eval.py both call `run_evaluation`.
"""

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from quantlab.config import PROJECT_ROOT, Settings
from quantlab.data import Catalog, load_catalog
from quantlab.guardrails import Action, Issue, effective_params, fill_defaults, validate_plan
from quantlab.mock_scenarios import SCENARIOS_BY_ID
from quantlab.plan import Plan
from quantlab.planner import generate_plan
from quantlab.registry import REGISTRY

GOLDEN_PATH = PROJECT_ROOT / "eval" / "golden_requests.yaml"
REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"

# The keys of a golden case's `guardrails` entry (one per action), and the item
# names they are reported under.
GUARDRAIL_ITEMS = {
    "approve": (Action.APPROVE, "approvals"),
    "warn": (Action.WARN, "warnings"),
    "block": (Action.BLOCK, "blocks"),
}


@dataclass
class GoldenCase:
    """One golden request (see the YAML file header for the fields)."""

    id: str
    request: str
    datasets: list[str]
    expect: str
    capabilities: list[str] = field(default_factory=list)
    key_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    guardrails: dict[str, list[str]] | None = None


@dataclass
class ItemCheck:
    """One scored item of a golden case: what was expected and what the planner chose.

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
    """How the planner did on one golden case.

    Fields:
        case_id: the case.
        expect: "plan" or "refuse_or_block".
        planned: capability sequence the planner produced ([] if none).
        issue_codes: validation issue codes.
        valid: plan exists, is not a refusal and has no BLOCK.
        refused_or_blocked: no plan, a refusal, or a BLOCK.
        sequence_match: capability sequence equals the expected one.
        jaccard: capability-set overlap with the expected set.
        params_hit: expected parameter values found.
        params_total: expected parameter values checked.
        correct: valid + exact sequence (+ the expected guardrail outcome, where given)
            for normal cases; refused/blocked for adversarial ones.
        error: planner error, if any.
        guardrails_match: issues are exactly the expected ones (None if not specified).
        checks: the item-by-item comparison (normal cases, and adversarial cases that
            give an expected outcome).
    """

    case_id: str
    expect: str
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


@dataclass
class EvalReport:
    """Aggregate result of an evaluation run."""

    mode: str
    model: str
    created_at: str
    scores: list[CaseScore]
    metrics: dict[str, float | None]


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenCase]:
    """Read the golden requests, resolving `scenario:` references to their request text.

    Args:
        path: the YAML file.

    Returns:
        The cases, in file order.

    Raises:
        ValueError: if a case's `guardrails` entry has a key other than block,
            approve or warn.
    """
    cases = []
    for item in yaml.safe_load(path.read_text(encoding="utf-8")):
        if "scenario" in item:
            scenario = SCENARIOS_BY_ID[item["scenario"]]
            request, datasets = scenario.request, list(scenario.dataset_ids)
        else:
            request, datasets = item["request"], item["datasets"]
        unknown = sorted(set(item.get("guardrails") or {}) - set(GUARDRAIL_ITEMS))
        if unknown:
            # A misspelt key would otherwise be silently ignored, and pass as "none".
            raise ValueError(f"{item['id']}: unknown guardrails key(s) {unknown}; "
                             f"use {list(GUARDRAIL_ITEMS)}.")  # fmt: skip
        cases.append(
            GoldenCase(
                id=item["id"],
                request=request,
                datasets=datasets,
                expect=item["expect"],
                capabilities=item.get("capabilities", []),
                key_params=item.get("key_params", {}),
                guardrails=item.get("guardrails"),
            )
        )
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
    case: GoldenCase, settings: Settings, catalog: Catalog, mode: str, client: Any = None
) -> CaseScore:
    """Plan one golden request, validate it and score it.

    Args:
        case: the golden case.
        settings: settings.
        catalog: dataset catalog.
        mode: "mock", "cached" or "live".
        client: optional Anthropic client (tests inject a fake one).

    Returns:
        The CaseScore.
    """
    result = generate_plan(case.request, case.datasets, mode, settings, catalog, client)
    plan = fill_defaults(result.plan) if result.plan else None
    report = validate_plan(plan, catalog, max_steps=settings.max_steps) if plan else None

    planned = [step.capability for step in plan.steps] if plan else []
    codes = report.codes() if report else sorted({i.code for i in result.issues})
    refused = plan is None or plan.cannot_fulfill
    blocked = report is not None and report.is_blocked
    valid = not refused and not blocked

    expected_set, planned_set = set(case.capabilities), set(planned)
    union = expected_set | planned_set
    jaccard = len(expected_set & planned_set) / len(union) if union else 1.0

    params = param_checks(case.key_params, plan)
    guardrails_match = None
    checks: list[ItemCheck] = []
    if case.expect == "plan":
        checks = sequence_checks(case.capabilities, planned) + params
    if case.guardrails is not None:
        # An adversarial case with an expected outcome (an example workflow) is reported
        # item by item too, for information: it stays correct if refused or blocked for
        # another reason, e.g. a live planner refusing what the canned plan lets the
        # validator block.
        if case.expect != "plan":
            checks = sequence_checks(case.capabilities, planned)
        outcome = guardrail_checks(case.guardrails, report.issues if report else result.issues)
        guardrails_match = all(check.match for check in outcome)
        checks += outcome

    sequence_match = planned == case.capabilities
    if case.expect == "plan":
        correct = valid and sequence_match and guardrails_match is not False
    else:
        correct = refused or blocked
    return CaseScore(
        case_id=case.id,
        expect=case.expect,
        planned=planned,
        issue_codes=codes,
        valid=valid,
        refused_or_blocked=refused or blocked,
        sequence_match=sequence_match,
        jaccard=round(jaccard, 4),
        params_hit=sum(check.match for check in params),
        params_total=len(params),
        correct=correct,
        error=result.error,
        guardrails_match=guardrails_match,
        checks=checks,
    )


def summarise(scores: list[CaseScore]) -> dict[str, float | None]:
    """Aggregate the per-case scores into the headline metrics (None when not applicable)."""
    normal = [s for s in scores if s.expect == "plan"]
    adversarial = [s for s in scores if s.expect == "refuse_or_block"]

    def rate(values: list[bool]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    params_total = sum(s.params_total for s in normal)
    return {
        "valid_plan_rate": rate([s.valid for s in normal]),
        "exact_sequence_match": rate([s.sequence_match for s in normal]),
        "mean_jaccard": round(sum(s.jaccard for s in normal) / len(normal), 4) if normal else None,
        "key_param_accuracy": (
            round(sum(s.params_hit for s in normal) / params_total, 4) if params_total else None
        ),
        "guardrail_outcome_match": rate(
            [s.guardrails_match for s in normal if s.guardrails_match is not None]
        ),
        "correct_refusal_block_rate": rate([s.refused_or_blocked for s in adversarial]),
        "overall_correct": rate([s.correct for s in scores]),
    }


def run_evaluation(
    mode: str,
    settings: Settings,
    golden_path: Path = GOLDEN_PATH,
    case_ids: list[str] | None = None,
) -> EvalReport:
    """Run the planner on the golden requests and compute the metrics.

    Args:
        mode: "mock", "cached" or "live".
        settings: settings.
        golden_path: the golden requests file.
        case_ids: evaluate only these cases (None = all of them).

    Returns:
        The EvalReport.

    Raises:
        ValueError: if a requested case id is not in the golden file.
    """
    catalog = load_catalog(settings.catalog_path)
    cases = load_golden(golden_path)
    if case_ids is not None:
        unknown = sorted(set(case_ids) - {case.id for case in cases})
        if unknown:
            raise ValueError(f"unknown golden case id(s): {', '.join(unknown)}")
        cases = [case for case in cases if case.id in case_ids]
    scores = [score_case(case, settings, catalog, mode) for case in cases]
    return EvalReport(
        mode=mode,
        model="mock" if mode == "mock" else settings.planner_model,
        created_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        scores=scores,
        metrics=summarise(scores),
    )


def _percent(value: float | None) -> str:
    """Format a rate as a percentage, or "n/a"."""
    return "n/a" if value is None else f"{value:.0%}"


def write_report(report: EvalReport, reports_dir: Path = REPORTS_DIR) -> Path:
    """Write the evaluation as a Markdown report named after its timestamp.

    Args:
        report: the evaluation result.
        reports_dir: output folder.

    Returns:
        Path of the written file.
    """
    m = report.metrics
    lines = [
        f"# Planner evaluation ({report.mode})",
        "",
        f"- Created: {report.created_at}",
        f"- Planner: {report.model}",
        f"- Cases: {len(report.scores)}",
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
        f"| Valid-plan rate (normal cases) | {_percent(m['valid_plan_rate'])} |",
        f"| Exact capability-sequence match | {_percent(m['exact_sequence_match'])} |",
        f"| Mean capability-set Jaccard | {m['mean_jaccard']} |",
        f"| Key-parameter accuracy | {_percent(m['key_param_accuracy'])} |",
        f"| Guardrail-outcome match | {_percent(m['guardrail_outcome_match'])} |",
        f"| Correct refusal/block rate (adversarial) | "
        f"{_percent(m['correct_refusal_block_rate'])} |",
        f"| Overall correct | {_percent(m['overall_correct'])} |",
        "",
        "## Cases",
        "",
        "| Case | Expect | Correct | Valid | Sequence | Jaccard | Params | Issues | Planned |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in report.scores:
        params = f"{s.params_hit}/{s.params_total}" if s.params_total else "-"
        lines.append(
            f"| {s.case_id} | {s.expect} | {'yes' if s.correct else 'NO'} | "
            f"{'yes' if s.valid else 'no'} | {'yes' if s.sequence_match else 'no'} | "
            f"{s.jaccard} | {params} | {', '.join(s.issue_codes) or '-'} | "
            f"{' > '.join(s.planned) or '(none)'} |"
        )
    lines += ["", "## Item checks", ""]
    for s in report.scores:
        if not s.checks:
            continue
        lines += [f"### {s.case_id}", "", "| Item | Expected | Planned | Match |"]
        lines.append("|---|---|---|---|")
        lines += [
            f"| {c.item} | {c.expected} | {c.actual} | {'yes' if c.match else 'NO'} |"
            for c in s.checks
        ]
        lines.append("")
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path = reports_dir / f"{stamp}-{report.mode}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
