"""Analysis page: request -> plan -> validate -> edit/approve -> execute -> narrate -> trace.

All state lives in st.session_state.session (a quantlab.engine.Session). Each
button calls one engine method; the page then redraws from the session's state.
"""

import labels
import pandas as pd
import streamlit as st

from quantlab.config import LLM_MODES
from quantlab.engine import Session
from quantlab.guardrails import Action
from quantlab.lineage import replay
from quantlab.mock_scenarios import SCENARIOS, match_scenario
from quantlab.registry import REGISTRY

import ui

# The two decisions on a plan in their own colours: Execute green, Reject plan red.
# Streamlit has no colour option for a button, so each one has a key, which gives its
# container a CSS class, and these rules colour it while enabled (a disabled button
# keeps the neutral disabled look). White text on both: contrast 5.0:1 and 4.8:1.
DECISION_BUTTON_CSS = """
<style>
.st-key-execute button:not(:disabled) {
  background-color: #188038; border-color: #188038; color: #ffffff;
}
.st-key-execute button:not(:disabled):hover { background-color: #137333; border-color: #137333; }
.st-key-reject-plan button:not(:disabled) {
  background-color: #d93025; border-color: #d93025; color: #ffffff;
}
.st-key-reject-plan button:not(:disabled):hover {
  background-color: #b3261e; border-color: #b3261e;
}
</style>
"""

settings = ui.get_settings()
st.html(DECISION_BUTTON_CSS)  # style only: applied without taking space on the page
catalog = ui.get_catalog(settings)

for key, default in {**ui.analysis_defaults(), "llm_mode": settings.llm_mode}.items():
    st.session_state.setdefault(key, default)

# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Datasets")
    datasets = st.multiselect(
        "Datasets available to the planner",
        catalog.ids(),
        key="dataset_select",
        format_func=lambda d: ui.dataset_label(catalog, d),
        help="Full names of the available datasets:\n\n" + ui.dataset_names_help(catalog),
    )
    # Hovering a selected dataset's line shows its full name.
    for dataset_id in datasets:
        st.caption(ui.dataset_caption(catalog, dataset_id), unsafe_allow_html=True)

    st.header("LLM mode")
    # Switching mode starts a clean page (ui.reset_analysis). on_change fires only when
    # the value changes: with required=True, clicking the selected mode changes nothing.
    mode = st.segmented_control(
        "LLM mode",
        LLM_MODES,
        key="llm_mode",
        required=True,
        label_visibility="collapsed",
        on_change=ui.reset_analysis,
    )
    st.caption(
        f"Live and cached modes call **{settings.llm_provider}**: planner "
        f"`{settings.planner_model}`, narrator `{settings.narrator_model}`."
    )
    if mode != "mock" and not settings.has_api_key:
        st.warning(
            f"No {settings.api_key_name} is set: live and cached calls will fail unless a "
            "cached plan exists. Mock mode works offline.",
            icon=":material/key_off:",
        )

    # The example workflows run on canned plans, so they are offered in mock mode only.
    if mode == "mock":
        st.header("Example workflows")
        st.caption(
            "Ready-made requests: complete analyses (1, 5), a blocked request you can fix "
            "and re-run (2), and out-of-scope or unsafe requests QuantLab stops (3, 4)."
        )
        for scenario in SCENARIOS:
            st.button(
                scenario.label,
                key=f"scenario-{scenario.id}",
                on_click=ui.load_scenario,
                args=(scenario.id,),
                width="stretch",
            )

# ---------------------------------------------------------------- request

ui.render_header()
request = st.text_area("Request", key="request_text", height=110)
if st.button(
    "Generate plan", type="primary", icon=":material/auto_awesome:", disabled=not request.strip()
):
    with st.spinner("Planning..."):
        session = Session.start(request, datasets, mode, settings, interactive=True)
        session.create_plan()
    st.session_state.session = session
    st.session_state.replay_report = None

session: Session | None = st.session_state.session
if session is None:
    hint = "Type a request, then generate a plan."
    if mode == "mock":
        hint = "Type a request or pick an example workflow in the sidebar, then generate a plan."
    st.info(hint, icon=":material/lightbulb:")
    st.stop()

# ---------------------------------------------------------------- plan

st.divider()
st.markdown(f"Run `{session.run_id}` · {ui.STATUS_BADGE.get(session.status, session.status)}")
plan = session.plan
result = session.plan_result
# A completed or rejected run is final: its plan can no longer change.
finished = session.status in ("completed", "rejected")
# step id -> {param: value} for widget values that differ from the plan: changes
# the user has made but not applied (and re-validated) yet.
pending: dict[str, dict] = {}
# step id -> {param: value} for every parameter field, as it is now.
values: dict[str, dict] = {}

st.subheader("Plan")
# A canned plan that simulates a planner behaviour (e.g. one that follows an
# injection) says so, so it is never mistaken for what a real model produced.
simulated = match_scenario(session.request) if session.mode == "mock" else None
if simulated is not None and simulated.note:
    st.warning(simulated.note, icon=":material/smart_toy:")
if plan is None:
    st.error(result.error if result else "No plan.", title="The planner returned no usable plan")
elif plan.cannot_fulfill:
    st.error(ui.escape_md(plan.refusal_reason or ""), title="The planner declined this request")
else:
    st.info(
        "You can adjust parameters below. Every change is re-checked against the same "
        "controls before anything runs.",
        icon=":material/tune:",
    )
    if finished:
        st.caption(f"This run is {session.status}; generate a new plan to change it.")
    proposal = {step.id: step.params for step in session.proposal.steps}
    dry_run = session.report.dry_run if session.report else {}
    # Widget keys include the plan version, so after an edit is applied (or reset)
    # every widget starts again from the plan's current values.
    version = session.plan_hash[:12]
    issues = session.report.issues if session.report else []
    for position, step in enumerate(plan.steps, start=1):
        spec = REGISTRY.get(step.capability)
        # BLOCK and WARN findings badge and open a step; approval findings do not
        # change how a step is shown (they are handled in the Validation section).
        actions = {
            issue.action
            for issue in issues
            if issue.step_id == step.id and issue.action != Action.APPROVE
        }
        step_edited = any(
            not ui.same_value(value, proposal.get(step.id, {}).get(name))
            for name, value in step.params.items()
        )
        input_meta = dry_run.get(next(iter(step.inputs.values()), ""), {})
        summary = ui.step_summary(step, spec, input_meta.get("frequency"), catalog)
        # Collapsed unless the step has a BLOCK or WARN finding. Collapsing is visual only:
        # the editor below always runs, and approvals stay in the Validation section.
        # Keyed by plan version, so after an edit is applied every step starts again
        # from its default and a step with a new finding opens.
        with st.expander(
            ui.step_header(position, step, summary, actions, step_edited),
            expanded=bool(actions),
            key=f"step-{session.run_id}-{version}-{step.id}",
        ):
            inputs = ", ".join(f"{name} ← {ref}" for name, ref in step.inputs.items())
            predicted = dry_run.get(step.id)
            output = (
                f"{labels.data_type_label(predicted['data_type'])}, "
                f"{labels.kind_label(predicted['kind'])}, "
                f"{labels.frequency_label(predicted['frequency'])}"
                if predicted
                else "unknown"
            )
            st.caption(
                f"{step.id} · {labels.capability_title(step.capability)} · inputs: "
                f"{inputs or 'none'} · output (predicted): {output} · "
                f"{ui.escape_md(step.rationale)}"
            )
            if spec is None:
                st.caption("Not a registered capability: it can be neither run nor edited.")
                continue
            if not spec.params:
                st.caption("No parameters.")
                continue
            for spec_param, column in zip(spec.params, st.columns(len(spec.params)), strict=True):
                with column:
                    current = step.params.get(spec_param.name)
                    value = ui.param_widget(
                        spec_param,
                        current,
                        f"param-{session.run_id}-{version}-{step.id}-{spec_param.name}",
                        catalog,
                        disabled=finished,
                        capability=step.capability,
                    )
                    values.setdefault(step.id, {})[spec_param.name] = value
                    original = proposal.get(step.id, {}).get(spec_param.name)
                    if not ui.same_value(value, current):
                        st.caption(":orange[changed, not applied yet]")
                    elif not ui.same_value(current, original):
                        was = labels.param_value_label(step.capability, spec_param.name, original)
                        st.caption(f":blue[edited (was {ui.escape_md(was)})]")

    # Drawn after every field, so this run already sees each change, and a change set
    # back to the plan's value is no change.
    pending = ui.unapplied_edits(plan, values)
    edited = any(
        not ui.same_value(value, proposal.get(step.id, {}).get(name))
        for step in plan.steps
        for name, value in step.params.items()
    )
    left, right = st.columns(2)
    # Highlighted (primary) only while there are changes to apply, but enabled whenever
    # the plan can change: a number field sends its typed value only when it loses
    # focus, so a button disabled until then would swallow the first click (which only
    # moves the focus) and need a second. The click itself sends the value, and this run
    # already sees it in `pending`. The key keeps the button the same widget when its
    # style changes in that run: without one, Streamlit's widget id includes the type,
    # and the click on the white button would be lost.
    if left.button(
        "Apply changes & re-validate",
        key="apply-edits",
        type="primary" if pending else "secondary",
        icon=":material/rule:",
        disabled=finished,
        width="stretch",
    ):
        if pending:
            session.apply_param_edits(pending)
            st.session_state.flash = (
                "Changes applied and re-validated. Approvals were reset: they belong to the "
                "previous plan version, so approve this one again."
            )
        else:
            st.session_state.flash = "No changes to apply: edit a parameter above first."
        st.rerun()
    if right.button(
        "Reset to planner's proposal",
        icon=":material/undo:",
        disabled=finished or not edited,
        width="stretch",
    ):
        session.reset_to_proposal()
        st.session_state.flash = "Back to the planner's proposal; approvals were reset."
        st.rerun()

if plan is not None and plan.planner_notes:
    st.caption(
        f"Planner notes (shown for context, never trusted): {ui.escape_md(plan.planner_notes)}"
    )

# ---------------------------------------------------------------- validation

st.subheader("Validation")
if st.session_state.get("flash"):
    st.info(st.session_state.flash, icon=":material/info:")
    st.session_state.flash = None
report = session.report
ticked: list[str] = []
if report is not None:
    if not report.issues:
        st.success("No issues: the plan passed every validator guardrail.", icon=":material/check:")
    for action in (Action.BLOCK, Action.APPROVE, Action.WARN):
        for issue in [i for i in report.issues if i.action == action]:
            with st.container(border=True):
                ui.render_issue(issue, ui.step_titles(plan) if plan else None)
                if action == Action.APPROVE:
                    # Keyed by plan version: an edited plan starts with every box unticked.
                    is_ticked = st.checkbox(
                        "I approve this",
                        key=f"approve-{session.run_id}-{session.plan_hash[:12]}-{issue.issue_id}",
                        value=any(a.issue_id == issue.issue_id for a in session.approvals),
                        disabled=finished,
                    )
                    if is_ticked:
                        ticked.append(issue.issue_id)
    for approval in session.approvals:
        st.caption(
            f"Approved {approval.issue_id} at {approval.approved_at} by {approval.approved_by} "
            f"for plan version {approval.plan_hash[:12]}"
        )

# Execute is allowed only for an applied, unblocked, fully approved plan; say why not.
blocking = report.blocks if report else []
unticked = [i for i in report.approvals_required if i.issue_id not in ticked] if report else []
reason = ""
if finished:
    reason = ""
elif report is None or plan is None:
    reason = "There is no plan to execute."
elif plan.cannot_fulfill:
    reason = "The planner declined this request, so there is nothing to run."
elif pending:
    reason = "Apply or reset your parameter changes to continue."
elif blocking:
    reason = f"Resolve {len(blocking)} blocking issue{'' if len(blocking) == 1 else 's'} first."
elif unticked:
    reason = f"Resolve {len(unticked)} approval{'' if len(unticked) == 1 else 's'} to continue."
already_approved = {a.issue_id for a in session.approvals}
# Rejecting is the reviewer's other decision; a refusal or a missing plan has nothing to reject.
can_reject = not finished and plan is not None and not plan.cannot_fulfill
execute_col, reject_col = st.columns(2)
if execute_col.button(
    "Execute",
    key="execute",
    type="primary",
    icon=":material/play_arrow:",
    disabled=finished or bool(reason),
    width="stretch",
):
    session.approve([issue_id for issue_id in ticked if issue_id not in already_approved])
    with st.spinner("Executing and writing the narrative..."):
        session.execute()
        if session.status == "completed":
            session.narrate()
            session.save_record()
    st.rerun()
if reject_col.button(
    "Reject plan",
    key="reject-plan",
    icon=":material/block:",
    disabled=not can_reject,
    width="stretch",
):
    session.reject()
    st.rerun()
if reason:
    st.caption(reason)
if session.status == "rejected":
    st.info("You rejected this plan: nothing was run. Generate a new plan to start again.",
            icon=":material/block:")  # fmt: skip

# ---------------------------------------------------------------- results

execution = session.execution
if execution is None:
    st.stop()

st.subheader("Results")
if execution.error:
    st.error(ui.escape_md(execution.error), title=f"Execution {execution.status}")
artifacts = execution.artifacts
shading = ui.flags_by_input(plan.steps, artifacts)
records = {record.step_id: record for record in execution.step_records}
for step in plan.steps:
    record = records.get(step.id)
    if record is None:
        continue
    artifact = artifacts.get(step.id)
    key_output = artifact is not None and (
        artifact.data_type.value == "table" or step.id in shading or step.capability == "spread"
    )
    capability = ui.capability_label(step.capability, record.capability_code)
    label = f"{step.id} · {capability} · {record.status}"
    if record.n_obs is not None:
        label += f" · {record.n_obs:,} obs · {record.duration_ms:.0f} ms"
    with st.expander(label, expanded=key_output or record.status != "completed"):
        for issue in record.issues:
            ui.render_issue(issue)
        if record.error:
            st.error(ui.escape_md(record.error))
        if artifact is not None:
            ui.render_artifact(step.id, step.capability, artifact, shading.get(step.id), catalog)

# ---------------------------------------------------------------- narrative

narrative = session.narrative
# Step id -> capability, to label each fact with the step that produced it.
capabilities = {step.id: step.capability for step in plan.steps} if plan else {}
if narrative is not None:
    st.subheader("Narrative")
    # N01 UNGROUNDED_NARRATIVE: an LLM narrative quoting a number the run did not
    # produce is withheld; the issue lists the numbers, the checked template is shown.
    for issue in narrative.issues:
        with st.container(border=True):
            ui.render_issue(issue)
    grounding = narrative.grounding
    if grounding.grounded and not narrative.issues:  # after an N01, the issue says it all
        st.success(
            f"All {grounding.numbers_checked} numbers verified against computed results.",
            icon=":material/verified:",
        )
    st.markdown(ui.escape_md(narrative.text))
    if narrative.note:  # who wrote it and, only if N01 passed, that its numbers were checked
        st.caption(narrative.note)
    source = f"LLM ({narrative.model})" if narrative.source == "llm" else "deterministic template"
    st.caption(
        f"Written by: {source}" + (f" · LLM error: {narrative.error}" if narrative.error else "")
    )
    if narrative.withheld is not None:
        with st.expander("Withheld LLM narrative (not verified: failed N01)", expanded=False):
            st.caption(
                "Numbers that match no value of the run: "
                + ", ".join(narrative.withheld.grounding.ungrounded)
            )
            st.markdown(ui.escape_md(narrative.withheld.text))
    with st.expander("Facts sheet (the source of truth for every number)", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "fact": f.fact_id,
                        "what": labels.fact_label(
                            f.fact_id, capabilities.get(f.fact_id.split(".")[0])
                        ),
                        "value": f.display,
                    }
                    for f in narrative.facts
                ]
            ),
            hide_index=True,
            alt="Facts sheet",
        )

# ---------------------------------------------------------------- trace

executed = [step.id for step in plan.steps if step.id in artifacts]
if executed:
    st.subheader("Trace a result")
    st.caption(
        "Pick a step to see every step it depends on, down to the source file: "
        "capability, library call, parameters and output hash."
    )
    chosen_step = st.selectbox(
        "Step to trace",
        executed,
        index=len(executed) - 1,  # the final result by default (e.g. the test)
        format_func=lambda s: (
            f"{s} · {ui.capability_label(records[s].capability, records[s].capability_code)}"
        ),
    )
    ui.render_trace(session.trace(chosen_step), catalog)

# ---------------------------------------------------------------- run record

if session.manifest_path is not None:
    st.subheader("Run record")
    st.markdown(f"Manifest: `{session.manifest_path}`")
    left, right = st.columns(2)
    left.download_button(
        "Download manifest",
        session.manifest_path.read_bytes(),
        file_name=f"{session.run_id}-manifest.json",
        mime="application/json",
        icon=":material/download:",
    )
    if right.button("Replay this run (no LLM)", icon=":material/replay:"):
        st.session_state.replay_report = replay(session.run_id, settings)
    if st.session_state.replay_report is not None:
        ui.render_replay(st.session_state.replay_report, catalog)
