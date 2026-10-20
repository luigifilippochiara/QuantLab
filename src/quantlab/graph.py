"""Graph: a LangGraph prototype of the orchestrator, alongside engine.py (an experiment).

Role in the architecture: the same pipeline as `engine.Session`, written as a
LangGraph `StateGraph` instead of a dataclass with methods. Each node is a thin
wrapper around the function the engine calls: the planner, guardrails, executor,
narrator and lineage are imported unchanged, and this module never imports
engine.py. The nodes log the same monitoring events as the engine and save the
same run record, so a graph run can be replayed and traced like any other.

    START -> request_check -> planner -> validator -> approval -> executor -> narrator
          -> recorder -> END

    request_check  R01 refuses an oversized request before any LLM call
    planner        the planner proposes a plan (or refuses, P01); mock mode only here
    validator      the validator's guardrails; run again after every edit
    approval       interrupt(): the run pauses until a reviewer approves, edits or rejects
    executor       deterministic execution with the runtime guardrails
    narrator       narrative and grounding check (an ungrounded one is withheld, N01)
    recorder       saves the run record; every run ends here, blocked or not

Conditional edges: a BLOCK goes to the recorder and ends; open APPROVE issues go
to the approval interrupt; an edit goes back to the validator; a clean plan goes
to the executor. The state is checkpointed after every node by an in-memory
checkpointer, so a run paused at approval resumes with exactly the same state.
`run_graph` chains it all for headless use, like `engine.run_request`.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ConfigDict, Field

from quantlab.config import Settings, load_settings
from quantlab.data import Catalog, load_catalog
from quantlab.executor import StepRecord, execute_plan
from quantlab.guardrails import (
    Action,
    Category,
    GuardrailType,
    Issue,
    ValidationReport,
    check_edits,
    check_request_length,
    fill_defaults,
    validate_plan,
)
from quantlab.lineage import (
    Approval,
    LLMInfo,
    ParamEdit,
    RunRecord,
    dataset_versions,
    environment_info,
    new_run_id,
    param_changes,
    save_run,
)
from quantlab.monitoring import log_event, utc_now
from quantlab.narrator import NarrativeResult, narrate
from quantlab.plan import Plan, describe_plan, plan_hash
from quantlab.planner import PlanResult, generate_plan, planning_usage, unplanned_result
from quantlab.registry import registry_version
from quantlab.types import Artifact, DataType, Frequency, SeriesKind, TokenUsage


class GraphState(TypedDict, total=False):
    """State of one run, passed from node to node and checkpointed after each one.

    Fields:
        request: the user's request.
        dataset_ids: datasets the user selected.
        mode: LLM mode (always "mock" in this prototype).
        run_id: unique run id, also the checkpointer's thread id.
        created_at: UTC timestamp.
        interactive: True when a person reviews the plan; False for headless runs.
        plan_result: what the planner returned.
        plan_original: the plan exactly as the planner proposed it.
        plan: the current plan: defaults filled in, edits applied.
        report: the latest validation report.
        edit_issues: BLOCK issues of a refused edit, for the validator to add to its report.
        edits: parameter edits made by the reviewer.
        approvals: approvals granted for the current plan.
        review: the reviewer's last answer: approved, edited, rejected, not_approved,
            or refused (it named an issue that is not open; the reviewer is asked again).
        review_error: why the last answer was refused, shown when asking again.
        plan_shown_at: when the plan was first shown for review (None without one).
        decision: "approved" (executed) or "rejected", once decided.
        decided_at: when the decision was made.
        artifacts: step id -> output Artifact, for every step that completed.
        steps: one StepRecord per step the executor attempted.
        runtime_issues: the executor's issues (E codes, or the blocks that stopped it).
        execution_error: why execution did not complete, if it did not.
        narrative: the narrative, once written.
        record: the run record, once saved.
        manifest_path: where the run record was last saved.
        status: planned | ready | blocked | rejected | completed | failed.
        timings_ms: latency of each stage.
    """

    request: str
    dataset_ids: list[str]
    mode: str
    run_id: str
    created_at: str
    interactive: bool
    plan_result: PlanResult | None
    plan_original: Plan | None
    plan: Plan | None
    report: ValidationReport | None
    edit_issues: list[Issue]
    edits: list[ParamEdit]
    approvals: list[Approval]
    review: str | None
    review_error: str | None
    plan_shown_at: str | None
    decision: str | None
    decided_at: str | None
    artifacts: dict[str, Artifact]
    steps: list[StepRecord]
    runtime_issues: list[Issue]
    execution_error: str | None
    narrative: NarrativeResult | None
    record: RunRecord | None
    manifest_path: str | None
    status: str
    timings_ms: dict[str, float]


@dataclass
class GraphContext:
    """What every node needs but the checkpoint must not hold; passed again on resume.

    Fields:
        settings: settings in force.
        catalog: dataset catalog.
    """

    settings: Settings
    catalog: Catalog


class ReviewDecision(BaseModel):
    """The reviewer's answer to the approval interrupt (untrusted input, so validated).

    Fields:
        action: "approve" (run the plan with these approvals), "edit" (change
            parameter values, then validate again) or "reject" (do not run it).
        issue_ids: for approve, the APPROVE issues being approved.
        edits: for edit, step id -> {parameter name: new value}. Typed Any on
            purpose: check_edits judges its shape, as for an edit in the engine.
        approved_by: who decided ("user" in a UI, "cli" for the demo).
    """

    model_config = ConfigDict(extra="forbid")

    action: Literal["approve", "edit", "reject"]
    issue_ids: list[str] = Field(default_factory=list)
    edits: Any = None
    approved_by: str = "user"


# The classes the checkpointer may rebuild from a checkpoint: the types held in
# GraphState and the enums inside their dumps (an Issue's dump includes its
# computed type and category). A class missing here would come back as a plain
# dict or string, so tests/test_graph.py checks the whole state round-trips.
CHECKPOINT_TYPES = [
    PlanResult,
    Plan,
    Issue,
    Action,
    GuardrailType,
    Category,
    ValidationReport,
    ParamEdit,
    Approval,
    StepRecord,
    NarrativeResult,
    RunRecord,
    DataType,
    Frequency,
    SeriesKind,
    TokenUsage,
]


# ---------------------------------------------------------------------------
# Helpers: the engine's bookkeeping, as functions of the state
# ---------------------------------------------------------------------------


def _log(
    runtime: Runtime[GraphContext],
    state: GraphState,
    event_type: str,
    stage: str | None = None,
    latency_ms: float | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    """Append a monitoring event for this run."""
    log_event(
        runtime.context.settings.events_path,
        state["run_id"],
        event_type,
        stage=stage,
        latency_ms=latency_ms,
        details=details,
    )


def _issue_details(issue: Issue) -> dict[str, Any]:
    """The fields of an issue that go into a monitoring event."""
    return {
        "code": issue.code,
        "name": issue.name,
        "action": issue.action.value,
        "step_id": issue.step_id,
        "issue_id": issue.issue_id,
    }


def current_plan_hash(state: GraphState) -> str:
    """Version hash of the current plan; approvals are tied to it."""
    plan = state.get("plan")
    return plan_hash(plan) if plan else ""


def unresolved_approvals(state: GraphState) -> list[Issue]:
    """APPROVE issues of the current report that have not been approved yet."""
    report = state.get("report")
    if report is None:
        return []
    # Only approvals given for THIS plan version count (edits also clear them).
    current = current_plan_hash(state)
    approved = {a.issue_id for a in state.get("approvals", []) if a.plan_hash == current}
    return [i for i in report.approvals_required if i.issue_id not in approved]


def refusal_reason(state: GraphState) -> str:
    """Why the plan may not be executed now ("" if it may), worded as the engine words it."""
    plan, report = state.get("plan"), state.get("report")
    if plan is None or report is None:
        return "There is no valid plan."
    if state.get("decision") == "rejected":
        return "The plan was rejected."
    if report.is_rejected:
        codes = ", ".join(sorted({issue.code for issue in report.rejections}))
        return f"The plan was rejected ({codes}): write a new request."
    if report.is_blocked:
        return "The plan has blocking issues: " + ", ".join(i.issue_id for i in report.blocks)
    unresolved = unresolved_approvals(state)
    if unresolved:
        return "Approval required for: " + ", ".join(i.issue_id for i in unresolved)
    return ""


def warnings(state: GraphState) -> list[Issue]:
    """All WARN issues: from validation, from execution and from the narrative check."""
    report, narrative = state.get("report"), state.get("narrative")
    static = report.warnings if report else []
    runtime = [i for i in state.get("runtime_issues", []) if i.action == Action.WARN]
    return static + runtime + (narrative.issues if narrative else [])


def issue_codes(state: GraphState) -> list[str]:
    """Sorted codes of every issue raised in a run: request check, planner refusal,
    validator, executor and narrator check."""
    report, narrative = state.get("report"), state.get("narrative")
    codes = set(report.codes()) if report else set()
    codes |= {issue.code for issue in state.get("runtime_issues", [])}
    codes |= {issue.code for issue in narrative.issues} if narrative else set()
    return sorted(codes)


def _datasets_used(state: GraphState, catalog: Catalog) -> list[str]:
    """Dataset ids loaded by the plan, in order."""
    plan = state.get("plan")
    if plan is None:
        return []
    ids = [s.params.get("dataset_id") for s in plan.steps if s.capability == "load_series"]
    return [d for d in dict.fromkeys(ids) if isinstance(d, str) and catalog.get(d)]


def _error_message(state: GraphState) -> str | None:
    """Why the run did not complete (None if it completed or is still in progress)."""
    if state.get("status") not in ("blocked", "failed"):
        return None
    if state.get("execution_error"):
        return state["execution_error"]
    result = state.get("plan_result")
    if result is not None and result.error:
        return result.error
    return refusal_reason(state) or None


def build_record(state: GraphState, catalog: Catalog) -> RunRecord:
    """Assemble the run record from the state (without saving it), as the engine does."""
    result = state.get("plan_result")
    llm = None
    if result is not None:
        llm = LLMInfo(
            provider=result.provider,
            model=result.model,
            served_model=result.served_model,
            prompt_version=result.prompt_version,
            mode=state["mode"],
            from_cache=result.from_cache,
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            error=result.error,
        )
    timings = state.get("timings_ms", {})
    plan, original = state.get("plan"), state.get("plan_original")
    narrative = state.get("narrative")
    steps = state.get("steps", [])
    stage_tokens: dict[str, TokenUsage] = {}
    if "planning" in timings:  # the planner ran (R01 stops before it)
        stage_tokens["planning"] = planning_usage(result)
    if narrative is not None and narrative.usage is not None:
        stage_tokens["narration"] = narrative.usage
    edits = state.get("edits", [])
    # Edits are compared with the planner's proposal with defaults filled in.
    proposal = fill_defaults(original) if original else None
    return RunRecord(
        run_id=state["run_id"],
        created_at=state["created_at"],
        status=state["status"],
        request=state["request"],
        dataset_ids=state["dataset_ids"],
        llm=llm,
        plan_original=original.model_dump() if original else None,
        plan_final=plan.model_dump() if plan else None,
        plan_hash=current_plan_hash(state),
        edits=edits,
        param_diff=param_changes(proposal, plan, edits) if plan else [],
        validation=state.get("report"),
        approvals=state.get("approvals", []),
        datasets=dataset_versions(steps, catalog),
        environment=environment_info(),
        registry_version=registry_version(),
        steps=steps,
        warnings=warnings(state),
        runtime_issues=state.get("runtime_issues", []),
        narrative=narrative.model_dump() if narrative else None,
        stage_timings_ms=timings,
        stage_tokens=stage_tokens,
        error=_error_message(state),
        interactive=state.get("interactive", False),
        plan_shown_at=state.get("plan_shown_at"),
        decision=state.get("decision"),
        decided_at=state.get("decided_at"),
    )


def _save(runtime: Runtime[GraphContext], state: GraphState) -> tuple[RunRecord, Path]:
    """Save the manifest and outputs, and log the run's terminal event.

    Returns:
        The record and the path of its manifest.
    """
    record = build_record(state, runtime.context.catalog)
    path = save_run(record, state.get("artifacts", {}), runtime.context.settings.runs_dir)
    if record.status == "completed":
        _log(runtime, state, "run_completed", details={"n_steps": len(record.steps)})
    elif record.status in ("blocked", "failed"):
        _log(runtime, state, "run_failed", details={"status": record.status, "error": record.error})
    return record, path


def _planning_failed(
    runtime: Runtime[GraphContext], state: GraphState, issues: list[Issue]
) -> GraphState:
    """The update for a run that produced no plan at all (R01, or a planner error V01)."""
    for issue in issues:
        _log(runtime, state, "validation_issue", details=_issue_details(issue))
    return {"report": ValidationReport(issues=issues), "status": "blocked"}


# ---------------------------------------------------------------------------
# Nodes: each takes the state and returns the fields it changes
# ---------------------------------------------------------------------------


def request_check_node(state: GraphState, runtime: Runtime[GraphContext]) -> GraphState:
    """Log the request and refuse it if it is too long (R01), before any LLM call."""
    _log(
        runtime,
        state,
        "request_received",
        details={
            "request": state["request"][:300],
            "dataset_ids": state["dataset_ids"],
            "mode": state["mode"],
        },
    )
    too_long = check_request_length(state["request"])
    if not too_long:
        return {}
    result = unplanned_result(too_long, state["mode"], runtime.context.settings)
    return {"plan_result": result, **_planning_failed(runtime, state, too_long)}


def planner_node(state: GraphState, runtime: Runtime[GraphContext]) -> GraphState:
    """Ask the planner for a plan and fill in the registry's defaults."""
    settings, catalog = runtime.context.settings, runtime.context.catalog
    result = generate_plan(state["request"], state["dataset_ids"], state["mode"], settings, catalog)
    update: GraphState = {
        "plan_result": result,
        "timings_ms": {**state.get("timings_ms", {}), "planning": result.latency_ms},
    }
    details = {
        "mode": result.mode,
        "model": result.model,
        "from_cache": result.from_cache,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
    }
    if result.plan is None:
        _log(
            runtime,
            state,
            "plan_failed",
            stage="planning",
            latency_ms=result.latency_ms,
            details={**details, "error": result.error},
        )
        return {**update, **_planning_failed(runtime, state, result.issues)}

    _log(
        runtime,
        state,
        "plan_generated",
        stage="planning",
        latency_ms=result.latency_ms,
        details=details,
    )
    update["plan_original"] = result.plan
    update["plan"] = fill_defaults(result.plan)
    if not result.plan.cannot_fulfill:  # a refusal (P01) is not a plan to review
        update["plan_shown_at"] = utc_now()
    return update


def validator_node(state: GraphState, runtime: Runtime[GraphContext]) -> GraphState:
    """Validate the current plan: the same validator for planner output and edits."""
    settings = runtime.context.settings
    started = time.perf_counter()
    report = validate_plan(state["plan"], runtime.context.catalog, max_steps=settings.max_steps)
    latency = round((time.perf_counter() - started) * 1000, 2)
    _log(
        runtime,
        state,
        "plan_validated",
        stage="validation",
        latency_ms=latency,
        details={"codes": report.codes(), "blocked": report.is_blocked},
    )
    for issue in report.issues:
        _log(runtime, state, "validation_issue", details=_issue_details(issue))
    refused = state.get("edit_issues", [])
    if refused:
        # A refused edit changed nothing, but the plan stays blocked until a valid
        # edit: its BLOCKs lead the report, as in Session.apply_param_edits.
        report = ValidationReport(issues=refused + report.issues, dry_run=report.dry_run)
        for issue in refused:
            _log(runtime, state, "validation_issue", details=_issue_details(issue))
    update: GraphState = {
        "report": report,
        "edit_issues": [],
        "status": "blocked" if report.is_blocked else "ready",
        "timings_ms": {**state.get("timings_ms", {}), "validation": latency},
    }
    if update["status"] == "ready":
        # Save the plan awaiting a decision straight away, as Session.validate does,
        # so a run paused at the approval interrupt is on record too. A blocked run
        # is saved by the recorder, which logs its terminal event once.
        _, path = _save(runtime, {**state, **update})
        update["manifest_path"] = str(path)
    return update


def review_request(state: GraphState) -> dict[str, Any]:
    """The approval interrupt's payload: what the reviewer is asked to decide on."""
    return {
        "run_id": state["run_id"],
        "plan_hash": current_plan_hash(state),
        "plan": describe_plan(state["plan"]),
        "error": state.get("review_error"),
        "open_approvals": [
            {"issue_id": i.issue_id, "code": i.code, "step_id": i.step_id, "message": i.message}
            for i in unresolved_approvals(state)
        ],
    }


def approval_node(state: GraphState, runtime: Runtime[GraphContext]) -> GraphState:
    """Pause for the reviewer, then apply their decision: approve, edit or reject.

    LangGraph runs this node again from the top when the run resumes, so the
    interrupt comes before any side effect (an event logged earlier would be
    logged twice). On resume, `interrupt` returns the validated ReviewDecision;
    an answer that does not fit it raises pydantic's ValidationError, and the
    same interrupt can be answered again.

    An approval of an issue that is not open is refused without raising: once
    a node fails after its interrupt, LangGraph replays the same answer on every
    resume, so the run could never go on. The node asks again instead.
    """
    decision = interrupt(review_request(state), response_schema=ReviewDecision)
    if decision.action == "reject":
        decided_at = utc_now()
        _log(runtime, state, "plan_rejected", details={"plan_hash": current_plan_hash(state)})
        return {
            "review": "rejected",
            "review_error": None,
            "decision": "rejected",
            "decided_at": decided_at,
            "status": "rejected",
        }
    if decision.action == "edit":
        return {**_apply_edits(state, decision.edits), "review_error": None}

    open_issues = {issue.issue_id: issue for issue in unresolved_approvals(state)}
    issue_ids = list(dict.fromkeys(decision.issue_ids))
    # Check every id before recording any, so a bad id leaves no partial approval.
    unknown = [issue_id for issue_id in issue_ids if issue_id not in open_issues]
    if unknown:
        error = f"{', '.join(map(repr, unknown))}: not an open approval issue"
        return {"review": "refused", "review_error": error}
    approvals = list(state.get("approvals", []))
    for issue_id in issue_ids:
        issue = open_issues[issue_id]
        approvals.append(
            Approval(
                issue_id=issue_id,
                code=issue.code,
                step_id=issue.step_id,
                message=issue.message,
                approved_at=utc_now(),
                approved_by=decision.approved_by,
                plan_hash=current_plan_hash(state),
            )
        )
        _log(
            runtime,
            state,
            "approval_granted",
            details={"issue_id": issue_id, "code": issue.code, "approved_by": decision.approved_by},
        )
    update: GraphState = {"approvals": approvals, "review": "approved", "review_error": None}
    if unresolved_approvals({**state, **update}):
        # Not every approval was given: stop and record it; nothing is executed.
        update.update(review="not_approved", status="blocked")
    return update


def _apply_edits(state: GraphState, edits: Any) -> GraphState:
    """Change parameter VALUES on a copy of the plan; the validator judges the result.

    The plan's structure is locked: an edit that adds, removes or reorders steps,
    or touches anything but a registered parameter, is refused by check_edits and
    nothing is applied. Any change clears all approvals: they belong to the
    previous plan version.
    """
    rejected = check_edits(state["plan"], edits)
    if rejected:
        return {"review": "edited", "edit_issues": rejected}
    edited = state["plan"].model_copy(deep=True)
    steps = {step.id: step for step in edited.steps}
    changes = []
    for step_id, params in edits.items():
        for name, value in params.items():
            old = steps[step_id].params.get(name)
            # Compare types too: 12 -> 12.0 is a real change for the strict validator.
            if old == value and type(old) is type(value):
                continue
            steps[step_id].params[name] = value
            changes.append(
                ParamEdit(step_id=step_id, param=name, old=old, new=value, edited_at=utc_now())
            )
    if not changes:
        return {"review": "edited"}
    # An approval refers to the plan as it was; after any change, ask again.
    return {
        "review": "edited",
        "plan": edited,
        "edits": state.get("edits", []) + changes,
        "approvals": [],
    }


def executor_node(state: GraphState, runtime: Runtime[GraphContext]) -> GraphState:
    """Run the plan with the deterministic executor (the decision to run it).

    execute_plan validates the plan again and refuses to start if anything is
    blocked or unapproved, so this node cannot skip a check either.
    """
    decided_at = utc_now()
    plan = state["plan"]
    _log(runtime, state, "execution_started", details={"n_steps": len(plan.steps)})
    current = current_plan_hash(state)
    result = execute_plan(
        plan,
        runtime.context.catalog,
        {a.issue_id for a in state.get("approvals", []) if a.plan_hash == current},
        max_steps=runtime.context.settings.max_steps,
    )
    for record in result.step_records:
        event = "step_completed" if record.status == "completed" else "step_failed"
        _log(
            runtime,
            state,
            event,
            latency_ms=record.duration_ms,
            details={
                "step_id": record.step_id,
                "capability": record.capability,
                "status": record.status,
                "error": record.error,
            },
        )
    for issue in result.issues:
        _log(runtime, state, "guardrail_triggered", details=_issue_details(issue))
    _log(
        runtime,
        state,
        "execution_finished",
        stage="execution",
        latency_ms=result.duration_ms,
        details={"status": result.status},
    )
    return {
        # Executing is the decision to run the plan: its approval as a whole.
        "decision": "approved",
        "decided_at": decided_at,
        "artifacts": result.artifacts,
        "steps": result.step_records,
        "runtime_issues": result.issues,
        "execution_error": result.error,
        "status": result.status,
        "timings_ms": {**state.get("timings_ms", {}), "execution": result.duration_ms},
    }


def narrator_node(state: GraphState, runtime: Runtime[GraphContext]) -> GraphState:
    """Write the narrative for the completed run and check its numbers (N01)."""
    catalog = runtime.context.catalog
    approved_ids = {approval.issue_id for approval in state.get("approvals", [])}
    issues = warnings(state) + [
        i for i in state["report"].approvals_required if i.issue_id in approved_ids
    ]
    used = [catalog.entries[d] for d in _datasets_used(state, catalog)]
    # How an exchange rate is quoted, so the narrative reads its returns the right
    # way round (e.g. a fall in USD/CHF means the franc strengthened).
    quotes = {e.dataset_id: e.quote_convention for e in used if e.quote_convention}
    narrative = narrate(
        state["request"],
        state["plan"],
        state["artifacts"],
        issues,
        state["mode"],
        runtime.context.settings,
        extra_texts=[entry.title for entry in used],
        dataset_notes=quotes,
        data_sha256={entry.dataset_id: entry.sha256 for entry in used},
    )
    withheld = narrative.withheld
    # A withheld LLM narrative took time too: count it in the narration stage.
    latency = narrative.latency_ms + (withheld.latency_ms if withheld else 0.0)
    _log(
        runtime,
        state,
        "narrative_checked",
        stage="narration",
        latency_ms=latency,
        details={
            "source": narrative.source,
            "grounded": narrative.grounding.grounded,
            "ungrounded": narrative.grounding.ungrounded,
            "withheld_ungrounded": withheld.grounding.ungrounded if withheld else [],
            "error": narrative.error,
        },
    )
    for issue in narrative.issues:  # N01 UNGROUNDED_NARRATIVE
        _log(runtime, state, "guardrail_triggered", details=_issue_details(issue))
    return {
        "narrative": narrative,
        "timings_ms": {**state.get("timings_ms", {}), "narration": latency},
    }


def recorder_node(state: GraphState, runtime: Runtime[GraphContext]) -> GraphState:
    """Save the run record and outputs: the last node of every run."""
    record, path = _save(runtime, state)
    return {"record": record, "manifest_path": str(path)}


# ---------------------------------------------------------------------------
# Edges: where each node's result leads
# ---------------------------------------------------------------------------


def after_request_check(state: GraphState) -> str:
    """R01 ends the run; any other request goes to the planner."""
    return "recorder" if state.get("status") == "blocked" else "planner"


def after_planner(state: GraphState) -> str:
    """No plan at all (V01) ends the run; a plan or a refusal (P01) is validated."""
    return "validator" if state.get("plan") is not None else "recorder"


def after_validation(state: GraphState) -> str:
    """A BLOCK ends the run; open APPROVE issues pause it; a clean plan runs."""
    if state["report"].is_blocked:
        return "recorder"
    if unresolved_approvals(state):
        return "approval"
    return "executor"


def after_review(state: GraphState) -> str:
    """An edit is validated again; full approval runs the plan; a refused answer asks
    again; a rejection, or approval not given for every issue, ends the run."""
    if state["review"] == "edited":
        return "validator"
    if state["review"] == "approved":
        return "executor"
    if state["review"] == "refused":
        return "approval"
    return "recorder"


def after_execution(state: GraphState) -> str:
    """A completed run is narrated; a runtime BLOCK or a failure ends the run."""
    return "narrator" if state["status"] == "completed" else "recorder"


def new_checkpointer() -> InMemorySaver:
    """An in-memory checkpointer that can hold QuantLab's state.

    Pydantic models and dataclasses are stored as msgpack and rebuilt on load;
    the allowlist limits rebuilding to CHECKPOINT_TYPES. Artifacts hold pandas
    Series, which msgpack cannot encode, so that one value falls back to pickle.
    The checkpoints only ever live in this process's memory, so no pickle from
    outside is ever loaded.
    """
    serde = JsonPlusSerializer(pickle_fallback=True, allowed_msgpack_modules=CHECKPOINT_TYPES)
    return InMemorySaver(serde=serde)


def build_graph(checkpointer: InMemorySaver | None = None) -> CompiledStateGraph:
    """Wire the nodes and edges and compile the graph.

    Args:
        checkpointer: where to keep each run's state between nodes (default: a new
            in-memory one). Required by the approval interrupt.

    Returns:
        The compiled graph; one graph can hold many runs (one thread per run id).
    """
    builder = StateGraph(GraphState, context_schema=GraphContext)
    builder.add_node("request_check", request_check_node)
    builder.add_node("planner", planner_node)
    builder.add_node("validator", validator_node)
    builder.add_node("approval", approval_node)
    builder.add_node("executor", executor_node)
    builder.add_node("narrator", narrator_node)
    builder.add_node("recorder", recorder_node)

    builder.add_edge(START, "request_check")
    builder.add_conditional_edges("request_check", after_request_check, ["planner", "recorder"])
    builder.add_conditional_edges("planner", after_planner, ["validator", "recorder"])
    builder.add_conditional_edges(
        "validator", after_validation, ["approval", "executor", "recorder"]
    )
    builder.add_conditional_edges(
        "approval", after_review, ["validator", "executor", "approval", "recorder"]
    )
    builder.add_conditional_edges("executor", after_execution, ["narrator", "recorder"])
    builder.add_edge("narrator", "recorder")
    builder.add_edge("recorder", END)
    return builder.compile(checkpointer=checkpointer or new_checkpointer())


# ---------------------------------------------------------------------------
# Running the graph
# ---------------------------------------------------------------------------


@dataclass
class GraphRun:
    """Where a run stands when the graph returns: finished, or paused for review.

    Fields:
        config: the checkpointer config of the run (its thread id is the run id).
        context: settings and catalog, passed to the graph again on resume.
        state: the run's state so far.
        paused_for: the approval interrupt's payload while paused, else None.
    """

    config: dict[str, Any]
    context: GraphContext
    state: GraphState = field(default_factory=dict)
    paused_for: dict[str, Any] | None = None


def _invoke(graph: CompiledStateGraph, run: GraphRun, graph_input: Any) -> GraphRun:
    """Run the graph until it ends or pauses, and report where it stands."""
    output = graph.invoke(graph_input, run.config, context=run.context, version="v2")
    paused_for = output.interrupts[0].value if output.interrupts else None
    return GraphRun(run.config, run.context, output.value, paused_for)


def start_run(
    graph: CompiledStateGraph,
    request: str,
    dataset_ids: list[str],
    settings: Settings,
    mode: str = "mock",
    interactive: bool = False,
) -> GraphRun:
    """Start a run: it goes as far as it can, then ends or pauses at the approval interrupt.

    Args:
        graph: a graph from build_graph().
        request: the user's request.
        dataset_ids: datasets selected by the user.
        settings: settings in force.
        mode: LLM mode; the prototype accepts only "mock".
        interactive: True when a person reviews the plan.

    Returns:
        The GraphRun; `paused_for` is set while the run waits for a review.

    Raises:
        ValueError: for any mode other than mock.
    """
    if mode != "mock":
        # The prototype runs offline: it never calls an LLM API.
        raise ValueError(f"the LangGraph prototype runs in mock mode only, not {mode!r}")
    run_id = new_run_id()
    state: GraphState = {
        "request": request,
        "dataset_ids": list(dataset_ids),
        "mode": mode,
        "run_id": run_id,
        "created_at": utc_now(),
        "interactive": interactive,
        "status": "planned",
    }
    context = GraphContext(settings=settings, catalog=load_catalog(settings.catalog_path))
    run = GraphRun(config={"configurable": {"thread_id": run_id}}, context=context)
    return _invoke(graph, run, state)


def resume_run(graph: CompiledStateGraph, run: GraphRun, decision: dict[str, Any]) -> GraphRun:
    """Answer the approval interrupt of a paused run and let it go on from the checkpoint.

    Args:
        graph: the graph the run was started on (its checkpointer holds the state).
        run: the paused run.
        decision: the reviewer's answer, in ReviewDecision's fields, e.g.
            {"action": "approve", "issue_ids": ["V10:s3:return_method"]}.

    Returns:
        The GraphRun after the decision: finished, or paused again (after an edit).
    """
    return _invoke(graph, run, Command(resume=decision))


def run_graph(
    request: str,
    dataset_ids: list[str],
    approve: bool = False,
    settings: Settings | None = None,
    graph: CompiledStateGraph | None = None,
) -> GraphState:
    """Run the whole graph headlessly, like engine.run_request.

    Args:
        request: the user's request.
        dataset_ids: selected datasets.
        approve: approve every open APPROVE issue at the interrupt (else none is,
            and the run is recorded as blocked, nothing executed).
        settings: settings (default load_settings()).
        graph: a graph to run on (default: a new one).

    Returns:
        The final state. Its status says how far the run got: "blocked",
        "failed" or "completed".
    """
    graph = graph or build_graph()
    run = start_run(graph, request, dataset_ids, settings or load_settings())
    if run.paused_for is not None:
        open_ids = [issue["issue_id"] for issue in run.paused_for["open_approvals"]]
        decision = {"action": "approve", "issue_ids": open_ids if approve else []}
        run = resume_run(graph, run, {**decision, "approved_by": "cli"})
    return run.state
