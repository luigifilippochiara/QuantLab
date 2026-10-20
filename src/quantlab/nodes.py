"""Nodes: the state of one run and the nodes that change it.

Role in the architecture: the steps of the graph that graph.py wires. Each node is a
thin wrapper around one stage's function: guardrails (the request check), the
planner, validator, executor, narrator and lineage hold all the logic, and a node
only logs its stage's monitoring events and returns the fields of the state it
changes. The recording helpers build the run record from the state and save it
(lineage.save_run).

Two rules keep the approval node, where a run pauses for review, safe to resume
(approval_node explains both): the interrupt comes before any side effect, and an
answer that cannot be applied is refused, never raised, so the reviewer is asked
again.

The module reads top to bottom: the run's state, its context and the reviewer's
answer; reading the state (what the nodes, the Analysis page and the CLI ask of a
run); recording (monitoring events and the run record); the nodes; and the edge
after the validator, which validator_node consults too. Nothing here imports
graph.py.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.runtime import Runtime
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field

from quantlab.config import Settings
from quantlab.data import Catalog
from quantlab.executor import StepRecord, execute_plan
from quantlab.guardrails import Action, Issue, ValidationReport, check_request_length
from quantlab.lineage import (
    Approval,
    LLMInfo,
    ParamEdit,
    PlanRevision,
    RunRecord,
    dataset_versions,
    environment_info,
    param_changes,
    save_run,
)
from quantlab.monitoring import log_event, utc_now
from quantlab.narrator import NarrativeResult, narrate
from quantlab.plan import Plan, describe_plan, plan_hash
from quantlab.planner import (
    PlanResult,
    RevisionRequest,
    generate_plan,
    planning_usage,
    unplanned_result,
)
from quantlab.registry import registry_version
from quantlab.types import Artifact, TokenUsage
from quantlab.validator import check_edits, fill_defaults, validate_plan

# ---------------------------------------------------------------------------
# The run's state, its context and the reviewer's answer
# ---------------------------------------------------------------------------


# Change requests one run may make: each is a planner call, and a person asks for it.
MAX_CHANGE_REQUESTS = 3


class GraphState(TypedDict, total=False):
    """State of one run, passed from node to node and checkpointed after each one.

    Fields:
        request: the user's request.
        dataset_ids: datasets the user selected.
        mode: LLM mode: "mock", "live" or "cached" (change requests need "live").
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
            changes_requested, or refused (it could not be applied, e.g. it named an
            issue that is not open; the reviewer is asked again).
        review_error: why the last answer was refused, shown when asking again.
        revisions: every round of planning so far (revision 0, then one per change request).
        revision_request: an accepted change request, until the planner has answered it.
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
    revisions: list[PlanRevision]
    revision_request: RevisionRequest | None
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
        client: optional OpenAI or Anthropic client for the planner and the narrator
            (tests inject a fake one; None: the SDK client of the configured provider).
    """

    settings: Settings
    catalog: Catalog
    client: Any = None


class ReviewDecision(BaseModel):
    """The reviewer's answer to the approval interrupt (untrusted input, so validated).

    Fields:
        action: "approve" (run the plan with these approvals), "edit" (change
            parameter values, then validate again), "reject" (do not run it) or
            "request_changes" (send the plan back to the planner with a comment).
        issue_ids: for approve, the APPROVE issues being approved.
        edits: for edit, step id -> {parameter name: new value}. Typed Any on
            purpose: check_edits judges its shape, and refuses any that is not a
            mapping of registered parameters.
        comment: for request_changes, what the reviewer wants changed (untrusted:
            the request check judges it before the planner sees it).
        approved_by: who decided ("user" in a UI, "cli" for the demo).
    """

    model_config = ConfigDict(extra="forbid")

    action: Literal["approve", "edit", "reject", "request_changes"]
    issue_ids: list[str] = Field(default_factory=list)
    edits: Any = None
    comment: str = ""
    approved_by: str = "user"


# ---------------------------------------------------------------------------
# Reading the state (also used by the Analysis page and the CLI)
# ---------------------------------------------------------------------------


def current_plan_hash(state: GraphState) -> str:
    """Version hash of the current plan; approvals are tied to it."""
    plan = state.get("plan")
    return plan_hash(plan) if plan else ""


def proposal(state: GraphState) -> Plan | None:
    """The planner's latest proposal with defaults filled in: what edits are compared and
    reset to."""
    original = state.get("plan_original")
    return fill_defaults(original) if original else None


def revision_number(state: GraphState) -> int:
    """The current revision: 0 for the planner's first proposal, n after n change requests."""
    return max(len(state.get("revisions", [])) - 1, 0)


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
    """Why the plan may not be executed now ("" if it may): the reason the reviewer sees."""
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


def change_request_refusal(state: GraphState) -> str:
    """Why the reviewer cannot request changes now ("" if they can).

    Live mode only: mock mode has no canned revisions, and in cached mode a change
    request would silently become a live call. A plan rejected as a whole (V01, V02)
    ends the run, so it is never reviewed; a decided run is final; and a run may ask
    at most MAX_CHANGE_REQUESTS times.
    """
    report = state.get("report")
    if state.get("mode") != "live":
        return "Available in live mode."
    if state.get("decision") is not None:
        return f"The plan was already {state['decision']}."
    if report is None or report.is_rejected or not state.get("revisions"):
        return "There is no plan to revise: write a new request."
    if revision_number(state) >= MAX_CHANGE_REQUESTS:
        return (
            f"All {MAX_CHANGE_REQUESTS} change requests of this run are used: execute or reject "
            "the plan."
        )
    return ""


def reset_edits(state: GraphState) -> dict[str, dict[str, Any]]:
    """The edit that sets every parameter back to the planner's latest proposal.

    Sent as an ordinary edit, so it is validated like any other.
    Values are compared with their types: 12 -> 12.0 is a change for the strict validator.
    """
    proposed = {step.id: step.params for step in proposal(state).steps}
    edits = {
        step.id: {
            name: proposed[step.id].get(name)
            for name, value in step.params.items()
            if value != proposed[step.id].get(name)
            or type(value) is not type(proposed[step.id].get(name))
        }
        for step in state["plan"].steps
    }
    return {step_id: params for step_id, params in edits.items() if params}


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


# ---------------------------------------------------------------------------
# Recording: monitoring events and the run record
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


def _revision_edits(state: GraphState) -> list[ParamEdit]:
    """The parameter edits made to the current revision."""
    current = revision_number(state)
    return [edit for edit in state.get("edits", []) if edit.revision == current]


def _manual_changes(state: GraphState) -> int:
    """Parameters the reviewer has changed from the current proposal (its param_diff)."""
    plan, original = state.get("plan"), proposal(state)
    if plan is None or original is None:
        return 0
    return len(param_changes(original, plan, _revision_edits(state)))


def _sum_usage(usages: list[TokenUsage]) -> TokenUsage:
    """The tokens of several planner calls added up (unknown if any call's count is)."""
    inputs = [u.input_tokens for u in usages]
    outputs = [u.output_tokens for u in usages]
    return TokenUsage(
        model=usages[-1].model,  # the model of the last call, whose price applies
        input_tokens=None if None in inputs else sum(inputs),
        output_tokens=None if None in outputs else sum(outputs),
        saved_input_tokens=sum(u.saved_input_tokens for u in usages),
        saved_output_tokens=sum(u.saved_output_tokens for u in usages),
    )


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
    """Assemble the run record from the state (without saving it)."""
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
    revisions = state.get("revisions", [])
    stage_tokens: dict[str, TokenUsage] = {}
    if revisions:  # the planner ran (R01 stops before it): every call of the run counts
        stage_tokens["planning"] = _sum_usage([revision.usage for revision in revisions])
    if narrative is not None and narrative.usage is not None:
        stage_tokens["narration"] = narrative.usage
    edits = state.get("edits", [])
    # Edits are compared with the planner's latest proposal with defaults filled in.
    diff = param_changes(proposal(state), plan, _revision_edits(state)) if original else []
    if revisions:
        # The last revision is left by the run's end: what it changed is the param_diff.
        revisions = [
            *revisions[:-1],
            revisions[-1].model_copy(update={"manual_changes": len(diff)}),
        ]
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
        param_diff=diff,
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
        revisions=revisions,
        manual_edits=sum(revision.manual_changes for revision in revisions),
        change_requests=max(len(revisions) - 1, 0),
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
    """Log the request and refuse it if it is too long (R01), before any LLM call.

    A change request comes back through here: the request and the reviewer's comment
    get the same check, so an oversized comment never reaches the planner either; it
    is refused and the reviewer is asked again, the plan unchanged.
    """
    pending = state.get("revision_request")
    if pending is not None:
        too_long = check_request_length(state["request"]) + check_request_length(pending.comment)
        if not too_long:
            return {}
        for issue in too_long:
            _log(runtime, state, "validation_issue", details=_issue_details(issue))
        error = "The change request was not sent to the planner: " + " ".join(
            issue.message for issue in too_long
        )
        return {"revision_request": None, "review": "refused", "review_error": error}
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
    """Ask the planner for a plan (or a revision of it) and fill in the registry's defaults.

    Every call is recorded as a revision of the run: revision 0, then one per change
    request. A revised plan replaces the previous one whole, so its approvals and any
    refused edit are cleared: they belong to the previous plan.
    """
    settings, catalog = runtime.context.settings, runtime.context.catalog
    revision = state.get("revision_request")
    requested_at = utc_now() if revision is not None else None
    result = generate_plan(
        state["request"],
        state["dataset_ids"],
        state["mode"],
        settings,
        catalog,
        runtime.context.client,
        revision,
    )
    timings = state.get("timings_ms", {})
    entry = PlanRevision(
        revision=len(state.get("revisions", [])),
        comment=revision.comment if revision is not None else None,
        requested_at=requested_at,
        planned_at=utc_now(),
        plan=result.plan.model_dump() if result.plan else None,
        plan_hash=plan_hash(fill_defaults(result.plan)) if result.plan else "",
        planner_error=result.error,
        from_cache=result.from_cache,
        latency_ms=result.latency_ms,
        usage=planning_usage(result),
    )
    update: GraphState = {
        "plan_result": result,
        # The planning stage of the run: every planner call of it, added up.
        "timings_ms": {**timings, "planning": timings.get("planning", 0.0) + result.latency_ms},
        "revisions": [*state.get("revisions", []), entry],
        "revision_request": None,
        "approvals": [],
        "edit_issues": [],
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
        failed = _planning_failed(runtime, state, result.issues)
        # No plan to show or revise: a planner failure (V01) ends the run.
        update["revisions"][-1] = entry.model_copy(
            update={"validation": failed["report"], "status": "blocked"}
        )
        return {**update, **failed, "plan": None, "plan_original": None}

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
    # The first plan shown starts the approval time; a refusal (P01) is not a plan to
    # review, and a revision does not restart the clock.
    if not result.plan.cannot_fulfill and not state.get("plan_shown_at"):
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
        # edit: its BLOCKs lead the report.
        report = ValidationReport(issues=refused + report.issues, dry_run=report.dry_run)
        for issue in refused:
            _log(runtime, state, "validation_issue", details=_issue_details(issue))
    status = "blocked" if report.is_blocked else "ready"
    update: GraphState = {
        "report": report,
        "edit_issues": [],
        "status": status,
        "timings_ms": {**state.get("timings_ms", {}), "validation": latency},
    }
    revisions = state.get("revisions", [])
    if revisions and revisions[-1].validation is None:
        # A new proposal's first validation is its outcome in the revision history.
        latest = revisions[-1].model_copy(update={"validation": report, "status": status})
        update["revisions"] = [*revisions[:-1], latest]
    if status == "ready" or after_validation({**state, **update}) == "approval":
        # Save a plan awaiting a decision straight away, so a run paused for review is
        # on record too (a blocked one with its terminal event). A run that ends here
        # blocked is saved by the recorder, once.
        _, path = _save(runtime, {**state, **update})
        update["manifest_path"] = str(path)
    return update


def review_request(state: GraphState) -> dict[str, Any]:
    """The approval interrupt's payload: what the reviewer is asked to decide on."""
    return {
        "run_id": state["run_id"],
        "status": state["status"],
        "revision": revision_number(state),
        "plan_hash": current_plan_hash(state),
        "plan": describe_plan(state["plan"]),
        "error": state.get("review_error"),
        "change_request": change_request_refusal(state) or "available",
        "open_approvals": [
            {"issue_id": i.issue_id, "code": i.code, "step_id": i.step_id, "message": i.message}
            for i in unresolved_approvals(state)
        ],
    }


def approval_node(state: GraphState, runtime: Runtime[GraphContext]) -> GraphState:
    """Pause for the reviewer, then apply their decision: approve, edit, reject or
    request changes.

    LangGraph runs this node again from the top when the run resumes, so the
    interrupt comes before any side effect (an event logged earlier would be
    logged twice). On resume, `interrupt` returns the validated ReviewDecision;
    an answer that does not fit it raises pydantic's ValidationError, and the
    same interrupt can be answered again.

    An answer that cannot be applied (an issue that is not open, an execution while
    the plan is blocked, a change request outside live mode or past the cap...) is
    refused without raising: once a node fails after its interrupt, LangGraph replays
    the same answer on every resume, so the run could never go on. The node asks
    again instead, with the reason.
    """
    decision = interrupt(review_request(state), response_schema=ReviewDecision)
    if decision.action == "request_changes":
        return _request_changes(state, decision.comment)
    if decision.action == "reject" and state["plan"].cannot_fulfill:
        # A refusal (P01) is not a plan to decide on.
        return {"review": "refused", "review_error": "There is no plan to reject."}
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
    if state.get("interactive"):
        # A person's Execute: refused while the plan is blocked or an approval is
        # missing; nothing is recorded, and the reviewer is asked again.
        missing = [i.issue_id for i in open_issues.values() if i.issue_id not in issue_ids]
        reason = refusal_reason(state) if state["report"].is_blocked else ""
        if missing and not reason:
            reason = "Approval required for: " + ", ".join(missing)
        if reason:
            return {"review": "refused", "review_error": reason}
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


def _request_changes(state: GraphState, comment: str) -> GraphState:
    """Accept a change request (the request check and the planner come next), or refuse it.

    The revision being left keeps how many parameters the reviewer had changed in it.
    The planner gets the plan as the reviewer saw it (edits applied), its hash, the
    findings on it (codes and messages, never data) and the comment.
    """
    reason = change_request_refusal(state)
    if not reason and not comment.strip():
        reason = "Write what should change first."
    if reason:
        return {"review": "refused", "review_error": reason}
    revisions = list(state["revisions"])
    revisions[-1] = revisions[-1].model_copy(update={"manual_changes": _manual_changes(state)})
    pending = RevisionRequest(
        previous_plan=state["plan"],
        previous_plan_hash=current_plan_hash(state),
        issues=state["report"].issues,
        comment=comment,
    )
    return {
        "review": "changes_requested",
        "review_error": None,
        "revisions": revisions,
        "revision_request": pending,
    }


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
                ParamEdit(
                    step_id=step_id,
                    param=name,
                    old=old,
                    new=value,
                    edited_at=utc_now(),
                    revision=revision_number(state),
                )
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
        client=runtime.context.client,
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
# The edge after the validator: wired in graph.py, consulted by validator_node to
# save a run that will pause for review
# ---------------------------------------------------------------------------


def after_validation(state: GraphState) -> str:
    """A plan rejected as a whole (V01, V02) ends the run. A person reviews every other
    plan, blocked or not; headless, a BLOCK ends the run, open APPROVE issues pause it
    and a clean plan runs."""
    report = state["report"]
    if report.is_rejected:
        return "recorder"
    if state.get("interactive"):
        return "approval"
    if report.is_blocked:
        return "recorder"
    if unresolved_approvals(state):
        return "approval"
    return "executor"
