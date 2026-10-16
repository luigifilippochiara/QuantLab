"""Lineage: run records (manifests), replay and trace.

Role in the architecture: every run is saved as runs/<run_id>/manifest.json
plus one output file per step (CSV for series and flags, JSON for tables).
The manifest records everything needed to explain and reproduce the result:
the request, what the LLM proposed, what the user changed and approved, the
exact data versions (file hashes), the library versions, the registry version,
and a hash of every step's output.

- `replay(run_id)` re-runs the final plan WITHOUT any LLM, using the recorded
  approvals, after checking the data files still have the recorded hashes, and
  compares every step's output hash with the original.
- `trace(record, step_id)` answers "where does this number come from?" by
  walking the plan backwards from a step to the source files.
"""

import importlib.metadata
import platform
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from quantlab.config import PROJECT_ROOT, Settings
from quantlab.data import Catalog, file_sha256, load_catalog
from quantlab.executor import StepRecord, execute_plan
from quantlab.guardrails import Issue, ValidationReport
from quantlab.monitoring import log_event
from quantlab.plan import Plan
from quantlab.types import Artifact, DataType, canonical_text

# Library versions recorded in every manifest (fixed names, never user input).
RECORDED_PACKAGES = (
    "pandas",
    "numpy",
    "pydantic",
    "openai",
    "anthropic",
)


class LLMInfo(BaseModel):
    """How the plan was produced.

    Fields:
        provider: "openai", "anthropic" or "mock".
        model: model id requested (or "mock").
        served_model: model that actually answered (differs if a refusal fallback ran).
        prompt_version: planner prompt version.
        mode: LLM mode used: live, cached or mock.
        from_cache: True if the plan came from the plan cache.
        latency_ms: planner latency.
        input_tokens: prompt tokens, if reported.
        output_tokens: completion tokens, if reported.
        error: planner error, if any.
    """

    provider: str
    model: str
    served_model: str | None = None
    prompt_version: str
    mode: str
    from_cache: bool = False
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None


class DatasetVersion(BaseModel):
    """Exact version of a dataset used by a run.

    Fields:
        dataset_id: catalog id.
        fred_id: FRED series id.
        sha256: snapshot file hash.
        retrieved_at: when the snapshot was downloaded.
        source_url: where it was downloaded from.
        rows_used: observations loaded (missing values dropped).
        start: first date used.
        end: last date used.
    """

    dataset_id: str
    fred_id: str
    sha256: str
    retrieved_at: str
    source_url: str
    rows_used: int | None = None
    start: str | None = None
    end: str | None = None


class Approval(BaseModel):
    """A user's approval of one APPROVAL issue.

    Fields:
        issue_id: the approved issue's id, e.g. "G12:s2:return_method".
        code: guardrail code.
        step_id: step concerned.
        message: the exact issue text that was approved.
        approved_at: UTC timestamp.
        approved_by: who approved (v1 has no authentication: "user" or "cli").
        plan_hash: version hash of the plan the approval was given for.
    """

    issue_id: str
    code: str
    step_id: str | None
    message: str
    approved_at: str
    approved_by: str = "user"
    plan_hash: str = ""


class ParamEdit(BaseModel):
    """One parameter changed by the user before execution.

    Fields:
        step_id: the step edited.
        param: parameter name.
        old: value before the edit.
        new: value after the edit.
        edited_at: UTC timestamp.
    """

    step_id: str
    param: str
    old: Any
    new: Any
    edited_at: str


class ParamChange(BaseModel):
    """One parameter whose final value differs from the planner's proposal.

    Fields:
        step_id: the step.
        param: parameter name.
        original: the value the planner proposed (defaults filled in).
        final: the value that ran.
        edited_at: UTC timestamp of the last edit to this parameter.
    """

    step_id: str
    param: str
    original: Any
    final: Any
    edited_at: str


class RunRecord(BaseModel):
    """The manifest of one run (saved as runs/<run_id>/manifest.json).

    Fields:
        run_id: unique, time-sortable id.
        created_at: UTC timestamp.
        status: planned | blocked | completed | failed.
        request: the user's request.
        dataset_ids: datasets selected by the user.
        llm: how the plan was produced.
        plan_original: the plan exactly as the planner returned it.
        plan_final: the plan that was validated / executed (defaults filled, user edits applied).
        plan_hash: version hash of plan_final (approvals store the hash they were given for).
        edits: every parameter edit, in order (including edits later undone).
        param_diff: planner-proposed vs final value of each user-edited parameter.
        validation: the final static validation report.
        approvals: approvals granted, with timestamps.
        datasets: exact dataset versions used.
        environment: Python, platform, library versions, git commit.
        registry_version: hash of the capability registry summary.
        steps: one StepRecord per executed step.
        warnings: all WARN issues (static and runtime).
        runtime_issues: issues raised during execution (R01-R04).
        narrative: narrator output and grounding check (see narrator.NarrativeResult).
        stage_timings_ms: latency per stage.
        error: why the run did not complete, if it did not.
    """

    run_id: str
    created_at: str
    status: str = "planned"
    request: str
    dataset_ids: list[str] = Field(default_factory=list)
    llm: LLMInfo | None = None
    plan_original: dict[str, Any] | None = None
    plan_final: dict[str, Any] | None = None
    plan_hash: str = ""
    edits: list[ParamEdit] = Field(default_factory=list)
    param_diff: list[ParamChange] = Field(default_factory=list)
    validation: ValidationReport | None = None
    approvals: list[Approval] = Field(default_factory=list)
    datasets: list[DatasetVersion] = Field(default_factory=list)
    environment: dict[str, Any] = Field(default_factory=dict)
    registry_version: str = ""
    steps: list[StepRecord] = Field(default_factory=list)
    warnings: list[Issue] = Field(default_factory=list)
    runtime_issues: list[Issue] = Field(default_factory=list)
    narrative: dict[str, Any] | None = None
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


# ---------------------------------------------------------------------------
# Building and saving records
# ---------------------------------------------------------------------------


def new_run_id() -> str:
    """Return a unique, time-sortable run id like "20261014-192442-1a2b3c"."""
    return f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def _git(*args: str) -> str | None:
    """Run a read-only git command in the project folder; None if git is unavailable."""
    try:
        completed = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def environment_info() -> dict[str, Any]:
    """Describe the software environment of this run.

    Returns:
        Dict with python version, platform, versions of the numerical libraries,
        the quantlab version, the git commit and whether the working tree had
        uncommitted changes (None when not in a git checkout).
    """
    status = _git("status", "--porcelain")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in RECORDED_PACKAGES},
        "quantlab": importlib.metadata.version("quantlab"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
    }


def dataset_versions(steps: list[StepRecord], catalog: Catalog) -> list[DatasetVersion]:
    """Collect the exact dataset versions used by the executed load_series steps.

    Args:
        steps: step records of the run.
        catalog: the dataset catalog.

    Returns:
        One DatasetVersion per distinct dataset loaded, in plan order.
    """
    versions: dict[str, DatasetVersion] = {}
    for step in steps:
        if step.capability != "load_series" or step.output_hash is None:
            continue
        entry = catalog.entries[step.params["dataset_id"]]
        versions.setdefault(
            entry.dataset_id,
            DatasetVersion(
                dataset_id=entry.dataset_id,
                fred_id=entry.fred_id,
                sha256=entry.sha256,
                retrieved_at=entry.retrieved_at,
                source_url=entry.source_url,
                rows_used=step.n_obs,
                start=step.start,
                end=step.end,
            ),
        )
    return list(versions.values())


def param_changes(proposal: Plan, final: Plan, edits: list[ParamEdit]) -> list[ParamChange]:
    """Parameter-level diff between the planner's proposal and the plan that ran.

    Args:
        proposal: the planner's plan with defaults filled in.
        final: the current plan (same steps; only values can differ).
        edits: the edit history, for the time of the last edit to each parameter.

    Returns:
        One ParamChange per parameter whose final value differs, in plan order.
    """
    proposed = {step.id: step.params for step in proposal.steps}
    last_edit = {(e.step_id, e.param): e.edited_at for e in edits}
    changes = []
    for step in final.steps:
        for name, value in step.params.items():
            original = proposed.get(step.id, {}).get(name)
            if value == original and type(value) is type(original):
                continue
            changes.append(
                ParamChange(step_id=step.id, param=name, original=original, final=value,
                            edited_at=last_edit.get((step.id, name), ""))
            )  # fmt: skip
    return changes


def output_filename(step_id: str, capability: str, data_type: DataType) -> str:
    """File name of a step's saved output, e.g. "s4_rolling_volatility.csv"."""
    extension = "csv" if data_type in (DataType.SERIES, DataType.FLAGS) else "json"
    return f"{step_id}_{capability}.{extension}"


def save_run(record: RunRecord, artifacts: dict[str, Artifact], runs_dir: Path) -> Path:
    """Write the manifest and every step output to runs/<run_id>/.

    Each output file holds the artifact's canonical text, so its SHA-256 equals
    the output_hash in the manifest (anyone can check with `shasum -a 256`).

    Args:
        record: the run record.
        artifacts: step id -> output artifact.
        runs_dir: the runs folder.

    Returns:
        Path of the manifest file.
    """
    run_dir = runs_dir / record.run_id
    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    capabilities_by_step = {step.step_id: step.capability for step in record.steps}
    for step_id, artifact in artifacts.items():
        name = output_filename(step_id, capabilities_by_step[step_id], artifact.data_type)
        (outputs_dir / name).write_text(canonical_text(artifact), encoding="utf-8")
    manifest = run_dir / "manifest.json"
    manifest.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return manifest


def load_run(run_id: str, runs_dir: Path) -> RunRecord:
    """Load a run record from its manifest.

    Args:
        run_id: the run id.
        runs_dir: the runs folder.

    Returns:
        The RunRecord.

    Raises:
        FileNotFoundError: if the run does not exist.
    """
    path = runs_dir / run_id / "manifest.json"
    return RunRecord.model_validate_json(path.read_text(encoding="utf-8"))


def list_runs(runs_dir: Path) -> list[dict[str, Any]]:
    """Summaries of all saved runs, newest first.

    Args:
        runs_dir: the runs folder.

    Returns:
        Rows of {run_id, created_at, status, request, n_steps, mode}.
    """
    rows = []
    for manifest in runs_dir.glob("*/manifest.json"):
        try:
            record = RunRecord.model_validate_json(manifest.read_text(encoding="utf-8"))
        except ValueError:
            continue  # a corrupt manifest must not break the history page
        rows.append(
            {
                "run_id": record.run_id,
                "created_at": record.created_at,
                "status": record.status,
                "request": record.request,
                "n_steps": len(record.steps),
                "mode": record.llm.mode if record.llm else None,
            }
        )
    return sorted(rows, key=lambda row: row["run_id"], reverse=True)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


class DatasetCheck(BaseModel):
    """Recorded vs current hash of one dataset file."""

    dataset_id: str
    recorded_sha256: str
    current_sha256: str | None
    match: bool


class StepReplay(BaseModel):
    """Original vs replayed output hash of one step."""

    step_id: str
    capability: str
    original_hash: str | None
    replay_hash: str | None
    match: bool


class ReplayReport(BaseModel):
    """Result of replaying a run without the LLM.

    Fields:
        run_id: the replayed run.
        all_match: True if data hashes, status and every step output hash match.
        original_status: status recorded in the manifest.
        replay_status: status of the replay execution (None if not executed).
        datasets: per-dataset hash checks.
        steps: per-step hash comparison.
        message: one-line summary.
    """

    run_id: str
    all_match: bool
    original_status: str
    replay_status: str | None
    datasets: list[DatasetCheck]
    steps: list[StepReplay]
    message: str


def replay(run_id: str, settings: Settings) -> ReplayReport:
    """Re-execute a recorded run deterministically and compare output hashes.

    Steps: load the manifest; verify every recorded dataset hash against the
    current file; re-validate and execute plan_final with the recorded
    approvals (no LLM involved); compare each step's output hash.

    Args:
        run_id: the run to replay.
        settings: settings (paths and budgets).

    Returns:
        A ReplayReport (also logged as a replay_completed event).
    """
    record = load_run(run_id, settings.runs_dir)
    catalog = load_catalog(settings.catalog_path)

    checks = []
    for version in record.datasets:
        current = (
            file_sha256(catalog.path_for(version.dataset_id))
            if catalog.get(version.dataset_id)
            else None
        )
        checks.append(
            DatasetCheck(
                dataset_id=version.dataset_id,
                recorded_sha256=version.sha256,
                current_sha256=current,
                match=current == version.sha256,
            )
        )

    if record.plan_final is None:
        return _replay_report(record, checks, settings, None, [], "The run has no plan to replay.")
    if not all(check.match for check in checks):
        changed = [c.dataset_id for c in checks if not c.match]
        message = f"Dataset files changed since the run: {changed}."
        return _replay_report(record, checks, settings, None, [], message)

    plan = Plan.model_validate(record.plan_final)
    result = execute_plan(
        plan,
        catalog,
        {approval.issue_id for approval in record.approvals},
        max_steps=settings.max_steps,
        max_datasets=settings.max_datasets,
    )
    replayed = {step.step_id: step for step in result.step_records}
    comparisons = []
    for original in record.steps:
        again = replayed.get(original.step_id)
        replay_hash = again.output_hash if again else None
        comparisons.append(
            StepReplay(
                step_id=original.step_id,
                capability=original.capability,
                original_hash=original.output_hash,
                replay_hash=replay_hash,
                match=replay_hash == original.output_hash,
            )
        )
    n_matching = sum(c.match for c in comparisons)
    message = (
        f"{n_matching} of {len(comparisons)} step outputs match; replay status "
        f"{result.status}, original status {record.status}."
    )
    return _replay_report(record, checks, settings, result.status, comparisons, message)


def _replay_report(
    record: RunRecord,
    checks: list[DatasetCheck],
    settings: Settings,
    replay_status: str | None,
    steps: list[StepReplay],
    message: str,
) -> ReplayReport:
    """Assemble a ReplayReport, decide whether everything matched, and log the event.

    Everything matches only if the data hashes match, the replay reached the
    same status as the original, the same steps ran, and every output hash matches.
    """
    all_match = (
        replay_status == record.status
        and all(check.match for check in checks)
        and len(steps) == len(record.steps)
        and all(step.match for step in steps)
    )
    log_event(
        settings.events_path,
        record.run_id,
        "replay_completed",
        details={"all_match": all_match, "replay_status": replay_status, "message": message},
    )
    return ReplayReport(
        run_id=record.run_id,
        all_match=all_match,
        original_status=record.status,
        replay_status=replay_status,
        datasets=checks,
        steps=steps,
        message=message,
    )


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------


class TraceStep(BaseModel):
    """One step in a lineage chain.

    `edited` maps each user-edited parameter to the value the planner proposed;
    the other parameters are exactly as the planner proposed them.
    """

    step_id: str
    capability: str
    library_calls: str
    params: dict[str, Any]
    edited: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, str]
    output_hash: str | None
    n_obs: int | None
    start: str | None
    end: str | None


class TraceResult(BaseModel):
    """Lineage of one step: every step it depends on, and the source files.

    Fields:
        run_id: the run.
        step_id: the step traced.
        chain: the step and all its ancestors, from the step back to the loads.
        sources: the dataset versions at the root of the chain.
    """

    run_id: str
    step_id: str
    chain: list[TraceStep]
    sources: list[DatasetVersion]


def trace(record: RunRecord, step_id: str) -> TraceResult:
    """Walk the plan backwards from a step to its source datasets.

    Args:
        record: the run record.
        step_id: the step whose lineage is wanted.

    Returns:
        A TraceResult: for each step capability, library call, params, input
        steps and output hash; for each source the dataset id, FRED id, file
        hash, retrieval date and the date range used.

    Raises:
        KeyError: if the step was not executed in this run.
    """
    steps = {step.step_id: step for step in record.steps}
    if step_id not in steps:
        raise KeyError(f"step {step_id!r} was not executed in run {record.run_id}")

    # Depth-first walk over input references to collect every ancestor.
    needed: set[str] = set()
    to_visit = [step_id]
    while to_visit:
        current = to_visit.pop()
        if current not in needed:
            needed.add(current)
            to_visit.extend(steps[current].input_step_ids.values())

    # Present the chain from the traced step back towards the data (reverse plan order).
    ordered = [step for step in reversed(record.steps) if step.step_id in needed]
    edited: dict[str, dict[str, Any]] = {}
    for change in record.param_diff:
        edited.setdefault(change.step_id, {})[change.param] = change.original
    chain = [
        TraceStep(
            step_id=step.step_id,
            capability=step.capability,
            library_calls=step.library_calls,
            params=step.params,
            edited=edited.get(step.step_id, {}),
            inputs=step.input_step_ids,
            output_hash=step.output_hash,
            n_obs=step.n_obs,
            start=step.start,
            end=step.end,
        )
        for step in ordered
    ]
    loaded = {step.params["dataset_id"] for step in ordered if step.capability == "load_series"}
    sources = [version for version in record.datasets if version.dataset_id in loaded]
    return TraceResult(run_id=record.run_id, step_id=step_id, chain=chain, sources=sources)
