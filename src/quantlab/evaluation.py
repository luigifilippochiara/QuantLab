"""Planner evaluation: does the planner pick the right plan for known requests?

Role in the architecture: this measures PLANNER accuracy only. It runs the
planner (mock or live) on the golden requests in eval/golden_requests.yaml,
validates each plan, and scores it against the expected capability sequence
and key parameters. It does not execute anything: whether the computations
themselves are right is measured separately, by the unit tests.

Metrics (normal cases = expect "plan"; adversarial = expect "refuse_or_block"):
- valid-plan rate: normal cases whose plan has no BLOCK and is not a refusal;
- exact sequence match: capability list equals the expected list, in order;
- capability-set Jaccard: |expected AND planned| / |expected OR planned|, averaged;
- key-param accuracy: share of expected parameter values found in the plan;
- correct refusal/block rate: adversarial cases that were refused or blocked.

The CLI (`quantlab.cli eval`) and eval/run_eval.py both call `run_evaluation`.
"""

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from quantlab.config import PROJECT_ROOT, Settings
from quantlab.data import Catalog, load_catalog
from quantlab.guardrails import effective_params, fill_defaults, validate_plan
from quantlab.mock_scenarios import SCENARIOS_BY_ID
from quantlab.planner import generate_plan
from quantlab.registry import REGISTRY

GOLDEN_PATH = PROJECT_ROOT / "eval" / "golden_requests.yaml"
REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"


@dataclass
class GoldenCase:
    """One golden request (see the YAML file header for the fields)."""

    id: str
    request: str
    datasets: list[str]
    expect: str
    capabilities: list[str] = field(default_factory=list)
    key_params: dict[str, dict[str, Any]] = field(default_factory=dict)


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
        correct: valid + exact sequence for normal cases; refused/blocked for adversarial ones.
        error: planner error, if any.
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
    """
    cases = []
    for item in yaml.safe_load(path.read_text(encoding="utf-8")):
        if "scenario" in item:
            scenario = SCENARIOS_BY_ID[item["scenario"]]
            request, datasets = scenario.request, list(scenario.dataset_ids)
        else:
            request, datasets = item["request"], item["datasets"]
        cases.append(
            GoldenCase(
                id=item["id"],
                request=request,
                datasets=datasets,
                expect=item["expect"],
                capabilities=item.get("capabilities", []),
                key_params=item.get("key_params", {}),
            )
        )
    return cases


def _same_value(expected: Any, actual: Any) -> bool:
    """Parameter equality: floats with a small tolerance, everything else exactly."""
    if isinstance(expected, float) or isinstance(actual, float):
        is_number = isinstance(actual, (int, float)) and not isinstance(actual, bool)
        return is_number and math.isclose(expected, actual, abs_tol=1e-9)
    return expected == actual


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

    hits = total = 0
    for capability, params in case.key_params.items():
        steps = [step for step in plan.steps if step.capability == capability] if plan else []
        for name, expected in params.items():
            total += 1
            # A hit only if EVERY step using the capability has the expected value
            # (e.g. if a plan resamples two series, both must be resampled weekly).
            if steps and all(
                _same_value(expected, effective_params(s, REGISTRY[capability]).get(name))
                for s in steps
            ):
                hits += 1

    sequence_match = planned == case.capabilities
    if case.expect == "plan":
        correct = valid and sequence_match
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
        params_hit=hits,
        params_total=total,
        correct=correct,
        error=result.error,
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
        "correct_refusal_block_rate": rate([s.refused_or_blocked for s in adversarial]),
        "overall_correct": rate([s.correct for s in scores]),
    }


def run_evaluation(mode: str, settings: Settings, golden_path: Path = GOLDEN_PATH) -> EvalReport:
    """Run the planner on every golden request and compute the metrics.

    Args:
        mode: "mock", "cached" or "live".
        settings: settings.
        golden_path: the golden requests file.

    Returns:
        The EvalReport.
    """
    catalog = load_catalog(settings.catalog_path)
    scores = [score_case(case, settings, catalog, mode) for case in load_golden(golden_path)]
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
            "> knows the five example workflows and a few canned requests, and refuses",
            "> everything else. Run with `--mode live` to measure the real planner.",
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
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path = reports_dir / f"{stamp}-{report.mode}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
