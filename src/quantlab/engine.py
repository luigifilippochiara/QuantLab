"""Engine: the single orchestrator used by the Streamlit app, the CLI and the notebooks.

Role in the architecture: a `Session` holds the state of one run and moves it
through the pipeline, in this order:

    Session.start(request, datasets, mode)
      .create_plan()          request check (R01) -> planner (LLM or mock) -> plan,
                              defaults filled -> validate; the plan is now shown
      .apply_param_edits()    user edits VALUES only -> re-validate with the SAME validator
      .reset_to_proposal()    back to the planner's proposal (re-validated)
      .approve(issue_ids)     user approves APPROVE issues (timestamped)
      .reject()               the reviewer decides not to run the plan (the run ends)
      .execute()              the decision to run it; deterministic executor (refuses
                              if blocked or unapproved)
      .narrate()              narrator + grounding check
      .save_record()          manifest + outputs under runs/<run_id>/

Every stage appends monitoring events, and the run record is saved whenever
the plan is validated, so a plan shown but never decided on is on record too.
The session is a plain dataclass, so its state is easy to inspect in a notebook
or in Streamlit's session_state.
`run_request` chains all stages for headless use (CLI, notebooks, eval).
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quantlab.config import Settings, load_settings
from quantlab.data import Catalog, load_catalog
from quantlab.executor import ExecutionResult, execute_plan
from quantlab.guardrails import (
    Action,
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
    TraceResult,
    dataset_versions,
    environment_info,
    new_run_id,
    param_changes,
    save_run,
    trace,
)
from quantlab.monitoring import log_event, utc_now
from quantlab.narrator import NarrativeResult, narrate
from quantlab.plan import Plan, plan_hash
from quantlab.planner import PlanResult, generate_plan, planning_usage, unplanned_result
from quantlab.registry import registry_version
from quantlab.types import TokenUsage


class ExecutionRefusedError(Exception):
    """Raised when execute() is called while the plan is blocked or not fully approved."""


class PlanRejectedError(ValueError):
    """Raised when a plan rejected as a whole (a terminal block: V01, V02) would be
    edited, re-validated, approved or rejected again: only a new request can go on."""


@dataclass
class Session:
    """State of one run, from request to saved record.

    Fields:
        request: the user's request.
        dataset_ids: datasets the user selected.
        mode: LLM mode ("live", "cached" or "mock").
        settings: settings in force.
        catalog: dataset catalog.
        run_id: unique run id.
        created_at: UTC timestamp.
        client: optional OpenAI or Anthropic client (tests inject a fake one).
        interactive: True when a person reviews the plan on the Analysis page;
            False for headless runs (CLI, notebooks, demo, tests).
        plan_result: what the planner returned.
        plan_original: the plan exactly as the planner proposed it.
        plan: the current plan: defaults filled in, user edits applied.
        report: the latest validation report.
        edits: parameter edits made by the user.
        approvals: approvals granted for the current plan.
        execution: the execution result, once executed.
        narrative: the narrative, once written.
        status: planned | ready | blocked | rejected | completed | failed.
        timings_ms: latency of each stage.
        manifest_path: where the run record was saved, once saved.
        plan_shown_at: when the plan was first shown for review (None without one).
        decision: "approved" (executed) or "rejected", once decided.
        decided_at: when the decision was made.
    """

    request: str
    dataset_ids: list[str]
    mode: str
    settings: Settings
    catalog: Catalog
    run_id: str
    created_at: str
    client: Any = None
    interactive: bool = False
    plan_result: PlanResult | None = None
    plan_original: Plan | None = None
    plan: Plan | None = None
    report: ValidationReport | None = None
    edits: list[ParamEdit] = field(default_factory=list)
    approvals: list[Approval] = field(default_factory=list)
    execution: ExecutionResult | None = None
    narrative: NarrativeResult | None = None
    status: str = "planned"
    timings_ms: dict[str, float] = field(default_factory=dict)
    manifest_path: Path | None = None
    plan_shown_at: str | None = None
    decision: str | None = None
    decided_at: str | None = None

    # ------------------------------------------------------------------ setup

    @classmethod
    def start(
        cls,
        request: str,
        dataset_ids: list[str],
        mode: str | None = None,
        settings: Settings | None = None,
        client: Any = None,
        interactive: bool = False,
    ) -> "Session":
        """Create a session for a new request.

        Args:
            request: the user's request.
            dataset_ids: datasets selected by the user.
            mode: LLM mode; defaults to the configured mode.
            settings: settings; defaults to load_settings().
            client: optional OpenAI or Anthropic client (tests).
            interactive: True when a person reviews the plan (the Analysis page).

        Returns:
            A new Session (a request_received event is logged).
        """
        settings = settings or load_settings()
        session = cls(
            request=request,
            dataset_ids=list(dataset_ids),
            mode=mode or settings.llm_mode,
            settings=settings,
            catalog=load_catalog(settings.catalog_path),
            run_id=new_run_id(),
            created_at=utc_now(),
            client=client,
            interactive=interactive,
        )
        session._log(
            "request_received",
            details={
                "request": request[:300],
                "dataset_ids": session.dataset_ids,
                "mode": session.mode,
            },
        )
        return session

    def _log(
        self,
        event_type: str,
        stage: str | None = None,
        latency_ms: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Append a monitoring event for this run."""
        log_event(
            self.settings.events_path,
            self.run_id,
            event_type,
            stage=stage,
            latency_ms=latency_ms,
            details=details,
        )

    # --------------------------------------------------------------- planning

    def create_plan(self) -> ValidationReport:
        """Ask the planner for a plan, fill in defaults and validate it.

        Returns:
            The validation report. If the request is too long, or the planner
            failed or refused, the report holds the R01, V01 or P01 issues and the
            run is recorded as blocked.
        """
        # R01 REQUEST_TOO_LONG: the request check, the first stage of the pipeline.
        too_long = check_request_length(self.request)
        if too_long:
            # Refuse before spending an LLM call on an oversized request.
            self.plan_result = unplanned_result(too_long, self.mode, self.settings)
            return self._planning_failed(too_long)

        result = generate_plan(
            self.request, self.dataset_ids, self.mode, self.settings, self.catalog, self.client
        )
        self.plan_result = result
        self.timings_ms["planning"] = result.latency_ms
        details = {
            "mode": result.mode,
            "model": result.model,
            "from_cache": result.from_cache,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        }
        if result.plan is None:
            self._log(
                "plan_failed",
                stage="planning",
                latency_ms=result.latency_ms,
                details={**details, "error": result.error},
            )
            return self._planning_failed(result.issues)

        self._log("plan_generated", stage="planning", latency_ms=result.latency_ms, details=details)
        self.plan_original = result.plan
        self.plan = fill_defaults(result.plan)
        if not result.plan.cannot_fulfill:  # a refusal (P01) is not a plan to review
            self.plan_shown_at = utc_now()
        return self.validate()

    def _planning_failed(self, issues: list[Issue]) -> ValidationReport:
        """Record a run that produced no plan at all (planner error or a request R01 refused)."""
        self.report = ValidationReport(issues=issues)
        for issue in issues:
            self._log("validation_issue", details=_issue_details(issue))
        self.status = "blocked"
        self.save_record()
        return self.report

    def validate(self) -> ValidationReport:
        """Validate the current plan (the same validator for planner output and user edits).

        Returns:
            The report. The run record is saved straight away, as blocked or as ready
            (awaiting a decision), so every plan shown is on record; when the run
            moves on, the same run id is overwritten.
        """
        if self.plan is None:
            raise ValueError("there is no plan to validate")
        started = time.perf_counter()
        self.report = validate_plan(
            self.plan,
            self.catalog,
            max_steps=self.settings.max_steps,
        )
        latency = round((time.perf_counter() - started) * 1000, 2)
        self.timings_ms["validation"] = latency
        self._log(
            "plan_validated",
            stage="validation",
            latency_ms=latency,
            details={"codes": self.report.codes(), "blocked": self.report.is_blocked},
        )
        for issue in self.report.issues:
            self._log("validation_issue", details=_issue_details(issue))
        self.status = "blocked" if self.report.is_blocked else "ready"
        self.save_record()
        return self.report

    # ----------------------------------------------------------- human review

    @property
    def proposal(self) -> Plan | None:
        """The planner's proposal with defaults filled in: what edits are compared and reset to."""
        return fill_defaults(self.plan_original) if self.plan_original else None

    @property
    def plan_hash(self) -> str:
        """Version hash of the current plan; approvals are tied to it."""
        return plan_hash(self.plan) if self.plan else ""

    def check_not_rejected(self) -> None:
        """Refuse any further action on a plan rejected as a whole.

        A terminal block (guardrails.Guardrail.fixable_by_edit False: V01, V02) means
        the plan's steps are wrong in kind or structure, which no parameter edit can
        change; the analyst writes a new request instead.

        Raises:
            PlanRejectedError: if the current report rejects the plan.
        """
        if self.report is not None and self.report.is_rejected:
            codes = ", ".join(sorted({issue.code for issue in self.report.rejections}))
            raise PlanRejectedError(f"The plan was rejected ({codes}): write a new request.")

    def apply_param_edits(self, edits: dict[str, dict[str, Any]]) -> ValidationReport:
        """Change parameter VALUES, then re-validate with the same validator.

        The plan's structure is locked: an edit that adds, removes or reorders
        steps, or touches anything but a registered parameter (capability, inputs,
        ids...), is rejected as a BLOCK and nothing is applied. Values are applied
        to a copy of the plan and then judged by the validator, so an out-of-range
        value is a V03 BLOCK even if the UI widgets should have prevented it. Any
        change clears all approvals: they belong to the previous plan version.

        Args:
            edits: step id -> {parameter name: new value}.

        Returns:
            The new validation report.

        Raises:
            ValueError: if there is no plan to edit.
            PlanRejectedError: if the plan was rejected as a whole.
        """
        if self.plan is None:
            raise ValueError("there is no plan to edit")
        self.check_not_rejected()
        rejected = check_edits(self.plan, edits)
        if rejected:
            # Nothing is applied; the plan stays blocked until a valid edit or a reset.
            report = self.validate()
            self.report = ValidationReport(issues=rejected + report.issues, dry_run=report.dry_run)
            for issue in rejected:
                self._log("validation_issue", details=_issue_details(issue))
            self.status = "blocked"
            self.save_record()
            return self.report

        edited = self.plan.model_copy(deep=True)
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
        if changes:
            self.plan = edited
            self.edits += changes
            # An approval refers to the plan as it was; after any change, ask again.
            self.approvals = []
        return self.validate()

    def reset_to_proposal(self) -> ValidationReport:
        """Undo every edit: set each parameter back to the planner's proposal, then re-validate.

        Raises:
            PlanRejectedError: if the plan was rejected as a whole.
        """
        self.check_not_rejected()
        proposal = {step.id: step.params for step in self.proposal.steps}
        edits = {
            step.id: {
                name: proposal[step.id].get(name)
                for name, value in step.params.items()
                if value != proposal[step.id].get(name)
                or type(value) is not type(proposal[step.id].get(name))
            }
            for step in self.plan.steps
        }
        return self.apply_param_edits({step_id: p for step_id, p in edits.items() if p})

    def unresolved_approvals(self) -> list[Issue]:
        """APPROVE issues of the current report that have not been approved yet."""
        if self.report is None:
            return []
        # Only approvals given for THIS plan version count (edits also clear them).
        approved = {a.issue_id for a in self.approvals if a.plan_hash == self.plan_hash}
        return [i for i in self.report.approvals_required if i.issue_id not in approved]

    def approve(self, issue_ids: list[str], approved_by: str = "user") -> None:
        """Record approval of specific APPROVE issues.

        Args:
            issue_ids: ids of the issues being approved (must be APPROVE issues
                of the current report).
            approved_by: who approved ("user" in the UI, "cli" for --approve-all).

        Raises:
            ValueError: if an id is not an open APPROVE issue.
            PlanRejectedError: if the plan was rejected as a whole.
        """
        self.check_not_rejected()
        open_issues = {issue.issue_id: issue for issue in self.unresolved_approvals()}
        for issue_id in issue_ids:
            if issue_id not in open_issues:
                raise ValueError(f"{issue_id!r} is not an open approval issue")
            issue = open_issues[issue_id]
            self.approvals.append(
                Approval(
                    issue_id=issue_id,
                    code=issue.code,
                    step_id=issue.step_id,
                    message=issue.message,
                    approved_at=utc_now(),
                    approved_by=approved_by,
                    plan_hash=self.plan_hash,
                )
            )
            self._log(
                "approval_granted",
                details={"issue_id": issue_id, "code": issue.code, "approved_by": approved_by},
            )

    def reject(self) -> None:
        """Record the reviewer's decision not to run the plan; the run ends here, unexecuted.

        Raises:
            ValueError: if there is no plan to decide on, or it was already decided.
            PlanRejectedError: if the validator already rejected the plan as a whole.
        """
        if self.plan is None or self.plan.cannot_fulfill:
            raise ValueError("there is no plan to reject")
        self.check_not_rejected()
        if self.decision is not None:
            raise ValueError(f"the plan was already {self.decision}")
        self.decision = "rejected"
        self.decided_at = utc_now()
        self.status = "rejected"
        self._log("plan_rejected", details={"plan_hash": self.plan_hash})
        self.save_record()

    def can_execute(self) -> tuple[bool, str]:
        """Whether execution is allowed now, and if not, why.

        Returns:
            (True, "") or (False, reason).
        """
        if self.plan is None or self.report is None:
            return False, "There is no valid plan."
        if self.decision == "rejected":
            return False, "The plan was rejected."
        if self.report.is_rejected:
            codes = ", ".join(sorted({issue.code for issue in self.report.rejections}))
            return False, f"The plan was rejected ({codes}): write a new request."
        if self.report.is_blocked:
            return False, "The plan has blocking issues: " + ", ".join(
                issue.issue_id for issue in self.report.blocks
            )
        unresolved = self.unresolved_approvals()
        if unresolved:
            return False, "Approval required for: " + ", ".join(i.issue_id for i in unresolved)
        return True, ""

    # -------------------------------------------------------------- execution

    def execute(self) -> ExecutionResult:
        """Run the plan with the deterministic executor.

        Returns:
            The ExecutionResult. A blocked or failed execution is saved immediately.

        Raises:
            ExecutionRefusedError: if the plan is blocked or approvals are missing.
        """
        allowed, reason = self.can_execute()
        if not allowed:
            raise ExecutionRefusedError(reason)
        # Executing is the decision to run the plan: its approval as a whole.
        self.decision = "approved"
        self.decided_at = utc_now()

        self._log("execution_started", details={"n_steps": len(self.plan.steps)})
        result = execute_plan(
            self.plan,
            self.catalog,
            {a.issue_id for a in self.approvals if a.plan_hash == self.plan_hash},
            max_steps=self.settings.max_steps,
        )
        self.execution = result
        self.timings_ms["execution"] = result.duration_ms
        for record in result.step_records:
            event = "step_completed" if record.status == "completed" else "step_failed"
            self._log(
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
            self._log("guardrail_triggered", details=_issue_details(issue))
        self._log(
            "execution_finished",
            stage="execution",
            latency_ms=result.duration_ms,
            details={"status": result.status},
        )
        self.status = result.status
        if result.status != "completed":
            self.save_record()
        return result

    def narrate(self) -> NarrativeResult:
        """Write and check the narrative for a completed execution.

        Returns:
            The NarrativeResult (LLM or template), with its grounding check.

        Raises:
            ValueError: if the run has not completed.
        """
        if self.execution is None or self.execution.status != "completed":
            raise ValueError("narrate() needs a completed execution")
        # A new narrative replaces any earlier one, and with it its N01 issues: the
        # narrator is told about the run's warnings, never about an old narrative.
        self.narrative = None
        approved_ids = {approval.issue_id for approval in self.approvals}
        issues = self.warnings() + [
            i for i in self.report.approvals_required if i.issue_id in approved_ids
        ]
        used = [self.catalog.entries[d] for d in self._datasets_used()]
        titles = [entry.title for entry in used]
        # How an exchange rate is quoted, so the narrative reads its returns the right
        # way round (e.g. a fall in USD/CHF means the franc strengthened).
        quotes = {e.dataset_id: e.quote_convention for e in used if e.quote_convention}
        narrative = narrate(
            self.request,
            self.plan,
            self.execution.artifacts,
            issues,
            self.mode,
            self.settings,
            extra_texts=titles,
            dataset_notes=quotes,
            client=self.client,
            data_sha256={entry.dataset_id: entry.sha256 for entry in used},
        )
        self.narrative = narrative
        withheld = narrative.withheld
        # A withheld LLM narrative took time too: count it in the narration stage.
        latency = narrative.latency_ms + (withheld.latency_ms if withheld else 0.0)
        self.timings_ms["narration"] = latency
        self._log(
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
            self._log("guardrail_triggered", details=_issue_details(issue))
        return narrative

    # ----------------------------------------------------------------- record

    def warnings(self) -> list[Issue]:
        """All WARN issues: from validation, from execution and from the narrative check."""
        static = self.report.warnings if self.report else []
        runtime = self.execution.issues if self.execution else []
        narrative = self.narrative.issues if self.narrative else []
        return static + [i for i in runtime if i.action == Action.WARN] + narrative

    def _datasets_used(self) -> list[str]:
        """Dataset ids loaded by the plan, in order."""
        if self.plan is None:
            return []
        ids = [s.params.get("dataset_id") for s in self.plan.steps if s.capability == "load_series"]
        return [d for d in dict.fromkeys(ids) if isinstance(d, str) and self.catalog.get(d)]

    def build_record(self) -> RunRecord:
        """Assemble the run record from the session's current state (without saving it)."""
        result = self.plan_result
        llm = None
        if result is not None:
            llm = LLMInfo(
                provider=result.provider,
                model=result.model,
                served_model=result.served_model,
                prompt_version=result.prompt_version,
                mode=self.mode,
                from_cache=result.from_cache,
                latency_ms=result.latency_ms,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                error=result.error,
            )
        steps = self.execution.step_records if self.execution else []
        stage_tokens: dict[str, TokenUsage] = {}
        if "planning" in self.timings_ms:  # the planner ran (R01 stops before it)
            stage_tokens["planning"] = planning_usage(result)
        if self.narrative is not None and self.narrative.usage is not None:
            stage_tokens["narration"] = self.narrative.usage
        return RunRecord(
            run_id=self.run_id,
            created_at=self.created_at,
            status=self.status,
            request=self.request,
            dataset_ids=self.dataset_ids,
            llm=llm,
            plan_original=self.plan_original.model_dump() if self.plan_original else None,
            plan_final=self.plan.model_dump() if self.plan else None,
            plan_hash=self.plan_hash,
            edits=self.edits,
            param_diff=param_changes(self.proposal, self.plan, self.edits) if self.plan else [],
            validation=self.report,
            approvals=self.approvals,
            datasets=dataset_versions(steps, self.catalog),
            environment=environment_info(),
            registry_version=registry_version(),
            steps=steps,
            warnings=self.warnings(),
            runtime_issues=self.execution.issues if self.execution else [],
            narrative=self.narrative.model_dump() if self.narrative else None,
            stage_timings_ms=self.timings_ms,
            stage_tokens=stage_tokens,
            error=self._error_message(),
            interactive=self.interactive,
            plan_shown_at=self.plan_shown_at,
            decision=self.decision,
            decided_at=self.decided_at,
        )

    def _error_message(self) -> str | None:
        """Why the run did not complete (None if it completed or is still in progress)."""
        if self.status not in ("blocked", "failed"):
            return None
        if self.execution is not None and self.execution.error:
            return self.execution.error
        if self.plan_result is not None and self.plan_result.error:
            return self.plan_result.error
        return self.can_execute()[1] or None

    def save_record(self) -> Path:
        """Save the manifest and outputs, and log the run's terminal event.

        Returns:
            Path of the manifest.
        """
        record = self.build_record()
        artifacts = self.execution.artifacts if self.execution else {}
        self.manifest_path = save_run(record, artifacts, self.settings.runs_dir)
        if record.status == "completed":
            self._log("run_completed", details={"n_steps": len(record.steps)})
        elif record.status in ("blocked", "failed"):
            self._log("run_failed", details={"status": record.status, "error": record.error})
        return self.manifest_path

    def trace(self, step_id: str) -> TraceResult:
        """Lineage of one executed step back to the source files (see lineage.trace)."""
        return trace(self.build_record(), step_id)


def _issue_details(issue: Issue) -> dict[str, Any]:
    """The fields of an issue that go into a monitoring event."""
    return {
        "code": issue.code,
        "name": issue.name,
        "action": issue.action.value,
        "step_id": issue.step_id,
        "issue_id": issue.issue_id,
    }


def run_request(
    request: str,
    dataset_ids: list[str],
    mode: str | None = None,
    approve: bool = False,
    settings: Settings | None = None,
    client: Any = None,
) -> Session:
    """Run the whole pipeline headlessly: plan, validate, (approve), execute, narrate, save.

    Args:
        request: the user's request.
        dataset_ids: selected datasets.
        mode: LLM mode (default from settings).
        approve: approve every APPROVE issue automatically (CLI --approve-all).
        settings: settings (default load_settings()).
        client: optional OpenAI or Anthropic client (tests).

    Returns:
        The finished Session. Its status says how far it got: "blocked" (static
        or runtime BLOCK, or approvals not given), "failed" or "completed".
    """
    session = Session.start(request, dataset_ids, mode, settings, client)
    report = session.create_plan()
    if report.is_blocked:
        return session
    if approve:
        session.approve([issue.issue_id for issue in session.unresolved_approvals()], "cli")
    if session.unresolved_approvals():
        # Not approved: stop here and record it, nothing is executed.
        session.status = "blocked"
        session.save_record()
        return session
    session.execute()
    if session.status == "completed":
        session.narrate()
        session.save_record()
    return session
