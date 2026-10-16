"""Deterministic executor: runs a validated plan step by step, with runtime guardrails.

Role in the architecture: this is the only place where capability functions are
called. For each step it looks the capability name up in REGISTRY (a plain
dict) and calls the pre-written function attached to it. There is no eval, no
exec, no getattr on modules: a name that is not a key of REGISTRY has no
function to call, so it cannot run.

The executor is also the last gate before computation. It re-validates the
plan with the same `validate_plan` the UI uses and refuses to start if any
BLOCK exists or any APPROVAL issue has not been approved, so calling it
directly (from a notebook, a test or replay) cannot skip the checks.

While running it applies the runtime guardrails that need data:
    R01 DATASET_HASH_MISMATCH (BLOCK)   before load_series
    R02 INSUFFICIENT_DATA     (BLOCK)   before any step with inputs, and after
                                        filter_period (the selected period)
    R03 MISSING_DATA          (WARN)    after load_series
    R04 ALIGNMENT_LOSS        (WARN)    before two-input steps
A BLOCK stops the run cleanly; the step records and issues so far are kept.
Each step produces a StepRecord: what ran, with which parameters, on which
inputs (by hash), what came out (by hash), how long it took.
"""

import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from quantlab.capabilities import inner_join
from quantlab.data import Catalog, verify_dataset_hash
from quantlab.guardrails import Issue, Severity, effective_params, make_issue, validate_plan
from quantlab.plan import Plan, Step
from quantlab.registry import REGISTRY, CapabilitySpec
from quantlab.types import Artifact, artifact_hash

# Runtime guardrail thresholds.
MISSING_DATA_THRESHOLD = 0.05  # R03: warn if more than 5% of a raw series is missing
ALIGNMENT_LOSS_THRESHOLD = 0.10  # R04: warn if a join drops more than 10% of an input


class CapabilityRefusedError(Exception):
    """Raised when asked to run a capability name that is not in the registry."""


class StepRecord(BaseModel):
    """What happened in one step (stored in the run record).

    Fields:
        step_id: the plan step id.
        capability: capability name.
        library_calls: the underlying library calls (from the registry).
        params: parameters actually used (defaults filled in).
        input_step_ids: input name -> step id it came from.
        input_hashes: input name -> hash of that input (for load_series:
            "dataset:<id>" -> snapshot file hash).
        output_hash: SHA-256 of the output's canonical text (None if not produced).
        output_type: DataType of the output.
        n_obs: observations in the output.
        start: first date of the output.
        end: last date of the output.
        meta: the output's metadata (kind, frequency, return method, notes...).
        duration_ms: wall-clock time of the step.
        status: "completed", "blocked" or "failed".
        issues: runtime guardrail issues raised by this step.
        error: error message if the step failed.
    """

    step_id: str
    capability: str
    library_calls: str
    params: dict[str, Any]
    input_step_ids: dict[str, str]
    input_hashes: dict[str, str]
    output_hash: str | None = None
    output_type: str | None = None
    n_obs: int | None = None
    start: str | None = None
    end: str | None = None
    meta: dict[str, Any] | None = None
    duration_ms: float = 0.0
    status: str = "completed"
    issues: list[Issue] = Field(default_factory=list)
    error: str | None = None


@dataclass
class ExecutionResult:
    """Outcome of executing a plan.

    Fields:
        status: "completed", "blocked" (a BLOCK guardrail or missing approval
            stopped it) or "failed" (a capability raised an error).
        artifacts: step id -> output Artifact, for every step that completed.
        step_records: one StepRecord per step that was attempted.
        issues: runtime issues (R01-R04), or the static blocks that prevented starting.
        error: human-readable reason when status is not "completed".
        duration_ms: total wall-clock time.
    """

    status: str
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    step_records: list[StepRecord] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    error: str | None = None
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def dispatch(step: Step, inputs: dict[str, Artifact], catalog: Catalog) -> Artifact:
    """Run one step by calling the registered function for its capability.

    Args:
        step: the plan step.
        inputs: input name -> Artifact produced by an earlier step.
        catalog: the dataset catalog (passed only to load_series).

    Returns:
        The step's output Artifact.

    Raises:
        CapabilityRefusedError: if the capability name is not a key of REGISTRY.
    """
    spec = REGISTRY.get(step.capability)
    if spec is None:
        # The whitelist by construction: no registry entry means no function.
        raise CapabilityRefusedError(f"'{step.capability}' is not a registered capability.")
    # e.g. compute_returns(series=<Artifact>, method="simple", periods=1)
    kwargs: dict[str, Any] = {**inputs, **effective_params(step, spec)}
    if spec.reads_catalog:
        kwargs["catalog"] = catalog
    return spec.func(**kwargs)


# ---------------------------------------------------------------------------
# Runtime guardrails
# ---------------------------------------------------------------------------


def check_dataset_hash(step: Step, catalog: Catalog) -> list[Issue]:
    """R01 DATASET_HASH_MISMATCH: the snapshot file must match the catalog's hash.

    Without it, a modified or replaced data file would silently produce results
    from data other than the recorded version, breaking reproducibility and audit.
    """
    dataset_id = step.params["dataset_id"]
    matches, actual = verify_dataset_hash(catalog, dataset_id)
    if matches:
        return []
    expected = catalog.entries[dataset_id].sha256
    return [
        make_issue(
            "R01",
            step.id,
            f"{dataset_id}: snapshot hash {actual[:12]}... differs from the catalog's "
            f"{expected[:12]}...; the file was changed.",
            "Restore the committed snapshot, or re-run scripts/fetch_data.py to create "
            "a new, recorded data version.",
        )
    ]


def _observations(spec: CapabilitySpec, inputs: dict[str, Artifact]) -> int:
    """Usable observations for a step: the input's length, or the size of the join."""
    if len(spec.inputs) == 1:
        return len(inputs[spec.inputs[0].name].value)
    first, second = (inputs[i.name].value for i in spec.inputs[:2])
    return len(inner_join(first, second, ("a", "b")))


def check_insufficient_data(
    step: Step, spec: CapabilitySpec, params: dict[str, Any], inputs: dict[str, Artifact]
) -> list[Issue]:
    """R02 INSUFFICIENT_DATA: enough observations for the capability's min_obs rule.

    Without it, statistics computed on a handful of points (a 52-week volatility
    from 60 weeks, an average of 3 returns) would be shown as findings when they
    are mostly noise.
    """
    if not spec.inputs:
        return []
    rule = spec.min_obs
    n_obs = _observations(spec, inputs)
    required = rule.min_obs
    if rule.window_param is not None:
        required = max(required, rule.window_multiple * params[rule.window_param])

    if n_obs >= required:
        return []
    return [
        make_issue(
            "R02",
            step.id,
            f"{spec.name} has {n_obs} usable observations; it needs {required} "
            f"({rule.description}).",
            "Use a longer history, a higher frequency, or a smaller window.",
        )
    ]


def check_insufficient_output(step: Step, spec: CapabilitySpec, output: Artifact) -> list[Issue]:
    """R02 INSUFFICIENT_DATA (output): enough observations left in a step's output.

    Only for capabilities whose min_obs rule sets output_min_obs (filter_period).
    Without it, a period that matches no data (e.g. after the last weekly close)
    would pass an empty series on, or end the run with an empty result.
    """
    required = spec.min_obs.output_min_obs
    if required is None or output.meta.n_obs >= required:
        return []
    return [
        make_issue(
            "R02",
            step.id,
            f"{spec.name} left {output.meta.n_obs} observations; at least {required} are "
            f"needed ({spec.min_obs.description}).",
            "Choose a longer period.",
            key="output",
        )
    ]


def check_missing_data(step: Step, output: Artifact) -> list[Issue]:
    """R03 MISSING_DATA: warn when more than 5% of a raw series was missing.

    Without it, a gappy series (e.g. a discontinued or illiquid series) could
    distort resampled values and returns, and nobody would know the data was thin.
    """
    dropped = output.meta.n_dropped_missing or 0
    total = output.meta.n_obs + dropped
    if total == 0 or dropped / total <= MISSING_DATA_THRESHOLD:
        return []
    return [
        make_issue(
            "R03",
            step.id,
            f"{dropped} of {total} raw values ({dropped / total:.1%}) were missing and "
            "have been dropped.",
            "Check the dataset's coverage before relying on the results.",
        )
    ]


def check_alignment_loss(
    step: Step, spec: CapabilitySpec, inputs: dict[str, Artifact]
) -> list[Issue]:
    """R04 ALIGNMENT_LOSS: warn when aligning two inputs drops more than 10% of either.

    Without it, an inner join of series with different histories (e.g. a yield
    starting in 1962 and one starting in 1976) silently discards a big part of
    the sample, changing what the result describes.
    """
    if len(spec.inputs) < 2:
        return []
    named = [(i.name, inputs[i.name].value) for i in spec.inputs[:2]]
    n_joined = len(inner_join(named[0][1], named[1][1], ("a", "b")))
    issues = []
    for name, series in named:
        lost = 1 - n_joined / len(series) if len(series) else 0.0
        if lost > ALIGNMENT_LOSS_THRESHOLD:
            issues.append(
                make_issue(
                    "R04",
                    step.id,
                    f"Aligning the inputs kept {n_joined} of {len(series)} rows of "
                    f"'{name}' ({lost:.0%} dropped).",
                    "Check that the inputs cover the same period; results describe only "
                    "the common dates.",
                    key=name,
                )
            )
    return issues


# ---------------------------------------------------------------------------
# Plan execution
# ---------------------------------------------------------------------------


def _elapsed_ms(start: float) -> float:
    """Milliseconds since `start` (a time.perf_counter() value), rounded to 0.01 ms."""
    return round((time.perf_counter() - start) * 1000, 2)


def execute_plan(
    plan: Plan,
    catalog: Catalog,
    approved_issue_ids: set[str] | None = None,
    *,
    max_steps: int = 15,
    max_datasets: int = 5,
) -> ExecutionResult:
    """Validate, then run every step of a plan in order.

    Args:
        plan: the plan (defaults may or may not be filled in).
        catalog: the dataset catalog.
        approved_issue_ids: issue ids of APPROVAL issues the user has approved.
        max_steps: budget passed to the validator.
        max_datasets: budget passed to the validator.

    Returns:
        An ExecutionResult. Never raises for problems in the plan or the data:
        they are reported through `status`, `issues` and `error`.
    """
    started = time.perf_counter()
    approved = approved_issue_ids or set()

    # Defence in depth: the same static checks the UI showed, re-run right here.
    report = validate_plan(plan, catalog, max_steps=max_steps, max_datasets=max_datasets)
    if report.is_blocked:
        return ExecutionResult(
            status="blocked",
            issues=report.blocks,
            error="The plan has blocking issues; nothing was run.",
            duration_ms=_elapsed_ms(started),
        )
    unresolved = [i for i in report.approvals_required if i.issue_id not in approved]
    if unresolved:
        return ExecutionResult(
            status="blocked",
            issues=unresolved,
            error="Approval required for: " + ", ".join(i.issue_id for i in unresolved),
            duration_ms=_elapsed_ms(started),
        )

    result = ExecutionResult(status="completed")
    hashes: dict[str, str] = {}  # step id -> output hash
    for step in plan.steps:
        step_started = time.perf_counter()
        spec = REGISTRY[step.capability]  # validated above, so the name exists
        params = effective_params(step, spec)
        inputs = {name: result.artifacts[ref] for name, ref in step.inputs.items()}
        record = StepRecord(
            step_id=step.id,
            capability=step.capability,
            library_calls=spec.library_calls,
            params=params,
            input_step_ids=dict(step.inputs),
            input_hashes={name: hashes[ref] for name, ref in step.inputs.items()},
        )
        if spec.reads_catalog:
            dataset_id = params["dataset_id"]
            record.input_hashes = {f"dataset:{dataset_id}": catalog.entries[dataset_id].sha256}

        # Runtime guardrails that must pass BEFORE the step runs.
        if spec.reads_catalog:
            record.issues += check_dataset_hash(step, catalog)
        record.issues += check_insufficient_data(step, spec, params, inputs)
        record.issues += check_alignment_loss(step, spec, inputs)
        blocking = [i for i in record.issues if i.severity == Severity.BLOCK]
        if blocking:
            record.status = "blocked"
            record.duration_ms = _elapsed_ms(step_started)
            result.step_records.append(record)
            result.issues += record.issues
            result.status = "blocked"
            result.error = f"Step {step.id} ({step.capability}): {blocking[0].message}"
            break

        try:
            output = dispatch(step, inputs, catalog)
        except Exception as exc:  # any error inside a capability stops the run cleanly
            record.status = "failed"
            record.error = f"{type(exc).__name__}: {exc}"
            record.duration_ms = _elapsed_ms(step_started)
            result.step_records.append(record)
            result.issues += record.issues
            result.status = "failed"
            result.error = f"Step {step.id} ({step.capability}) failed: {record.error}"
            break

        # Runtime guardrails that look at the output.
        if spec.reads_catalog:
            record.issues += check_missing_data(step, output)
        output_issues = check_insufficient_output(step, spec, output)
        if output_issues:
            record.issues += output_issues
            record.status = "blocked"
            record.duration_ms = _elapsed_ms(step_started)
            result.step_records.append(record)
            result.issues += record.issues
            result.status = "blocked"
            result.error = f"Step {step.id} ({step.capability}): {output_issues[0].message}"
            break

        hashes[step.id] = artifact_hash(output)
        record.output_hash = hashes[step.id]
        record.output_type = output.data_type.value
        record.n_obs = output.meta.n_obs
        record.start = output.meta.start
        record.end = output.meta.end
        record.meta = {
            "kind": output.meta.kind,
            "frequency": output.meta.frequency,
            "return_method": output.meta.return_method,
            "periods": output.meta.periods,
            "source_dataset_ids": output.meta.source_dataset_ids,
            "notes": output.meta.notes,
            # filter_period's resolved interval, kept in the run manifest.
            "period_start": output.meta.period_start,
            "period_end": output.meta.period_end,
        }
        record.duration_ms = _elapsed_ms(step_started)
        result.artifacts[step.id] = output
        result.step_records.append(record)
        result.issues += record.issues

    result.duration_ms = _elapsed_ms(started)
    return result
