"""Analysis page: request -> plan -> validate -> edit/approve -> execute -> narrate -> trace.

All state lives in st.session_state.session (a quantlab.engine.Session). Each
button calls one engine method; the page then redraws from the session's state.
"""

import json

import pandas as pd
import streamlit as st

from quantlab.config import LLM_MODES
from quantlab.engine import Session
from quantlab.lineage import replay
from quantlab.registry import REGISTRY
from quantlab.scenarios import SCENARIOS
from quantlab.validation import Severity

import ui

settings = ui.get_settings()
catalog = ui.get_catalog(settings)

for key, default in [
    ("session", None),
    ("request_text", ""),
    ("dataset_select", ["NASDAQCOM"]),
    ("llm_mode", settings.llm_mode),
    ("replay_report", None),
    ("flash", None),  # one-off message shown after an edit is applied
]:
    st.session_state.setdefault(key, default)

# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Datasets")
    datasets = st.multiselect(
        "Datasets available to the planner",
        catalog.ids(),
        key="dataset_select",
        format_func=lambda d: ui.dataset_name(catalog, d),
        help="Full names of the available datasets:\n\n" + ui.dataset_names_help(catalog),
    )
    # Hovering a selected dataset's line shows its full name.
    for dataset_id in datasets:
        st.caption(ui.dataset_caption(catalog, dataset_id), unsafe_allow_html=True)

    st.header("LLM mode")
    mode = st.segmented_control(
        "LLM mode", LLM_MODES, key="llm_mode", required=True, label_visibility="collapsed"
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
            "Ready-made requests: a typical volatility analysis (1) and the safeguards "
            "QuantLab applies to risky or out-of-scope requests (2-5)."
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
        session = Session.start(request, datasets, mode, settings)
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
finished = session.status == "completed"
# step id -> {param: value} for widget values that differ from the plan: changes
# the user has made but not applied (and re-validated) yet.
pending: dict[str, dict] = {}

st.subheader("Plan")
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
        st.caption("This run has completed; generate a new plan to change it.")
    proposal = {step.id: step.params for step in session.proposal.steps}
    dry_run = session.report.dry_run if session.report else {}
    # Widget keys include the plan version, so after an edit is applied (or reset)
    # every widget starts again from the plan's current values.
    version = session.plan_hash[:12]
    for step in plan.steps:
        spec = REGISTRY.get(step.capability)
        with st.container(border=True):
            inputs = ", ".join(f"{name} ← {ref}" for name, ref in step.inputs.items())
            predicted = dry_run.get(step.id)
            output = (
                " / ".join(str(predicted[k]) for k in ("data_type", "kind", "frequency"))
                if predicted
                else "unknown"
            )
            st.markdown(f"**{step.id} · {step.capability}**")
            st.caption(
                f"inputs: {inputs or 'none'} · output (predicted): {output} · "
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
                    )
                    original = proposal.get(step.id, {}).get(spec_param.name)
                    if not ui.same_value(value, current):
                        pending.setdefault(step.id, {})[spec_param.name] = value
                        st.caption(":orange[changed, not applied yet]")
                    elif not ui.same_value(current, original):
                        st.caption(f":blue[edited (was {original})]")

    edited = any(
        not ui.same_value(value, proposal.get(step.id, {}).get(name))
        for step in plan.steps
        for name, value in step.params.items()
    )
    left, right = st.columns(2)
    if left.button(
        "Apply changes & re-validate",
        type="primary",
        icon=":material/rule:",
        disabled=finished or not pending,
        width="stretch",
    ):
        session.apply_param_edits(pending)
        st.session_state.flash = (
            "Changes applied and re-validated. Approvals were reset: they belong to the "
            "previous plan version, so approve this one again."
        )
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
with st.expander("Raw JSON", icon=":material/data_object:"):
    st.markdown("Plan as proposed by the planner:")
    st.json(session.plan_original.model_dump() if session.plan_original else {}, expanded=False)
    st.markdown("Plan as it will run (defaults filled in, edits applied):")
    st.json(plan.model_dump() if plan else {}, expanded=False)
    st.markdown("Raw planner output:")
    st.code(json.dumps(result.raw_output if result else None, indent=2, default=str), "json")

# ---------------------------------------------------------------- validation

st.subheader("Validation")
if st.session_state.get("flash"):
    st.info(st.session_state.flash, icon=":material/info:")
    st.session_state.flash = None
report = session.report
ticked: list[str] = []
if report is not None:
    if not report.issues:
        st.success("No issues: the plan passed every static guardrail.", icon=":material/check:")
    for severity in (Severity.BLOCK, Severity.APPROVAL, Severity.WARN):
        for issue in [i for i in report.issues if i.severity == severity]:
            with st.container(border=True):
                ui.render_issue(issue)
                if severity == Severity.APPROVAL:
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
if st.button(
    "Execute", type="primary", icon=":material/play_arrow:", disabled=finished or bool(reason)
):
    session.approve([issue_id for issue_id in ticked if issue_id not in already_approved])
    with st.spinner("Executing and writing the narrative..."):
        session.execute()
        if session.status == "completed":
            session.narrate()
            session.save_record()
    st.rerun()
if reason:
    st.caption(reason)

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
    label = f"{step.id} · {step.capability} · {record.status}"
    if record.n_obs is not None:
        label += f" · {record.n_obs:,} obs · {record.duration_ms:.0f} ms"
    with st.expander(label, expanded=key_output or record.status != "completed"):
        for issue in record.issues:
            ui.render_issue(issue)
        if record.error:
            st.error(ui.escape_md(record.error))
        if artifact is not None:
            ui.render_artifact(step.id, step.capability, artifact, shading.get(step.id))

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
        index=len(executed) - 1,  # the final result by default (e.g. flag_regime)
        format_func=lambda s: f"{s} · {records[s].capability}",
    )
    ui.render_trace(session.trace(chosen_step), catalog)

# ---------------------------------------------------------------- narrative

narrative = session.narrative
if narrative is not None:
    st.subheader("Narrative")
    grounding = narrative.grounding
    if grounding.grounded:
        st.success(
            f"All {grounding.numbers_checked} numbers verified against computed results.",
            icon=":material/verified:",
        )
    else:
        st.error(
            "Ungrounded numbers: " + ", ".join(grounding.ungrounded) + ". Rely on the facts "
            "table below, which comes straight from the executor.",
            title="Numbers that could not be verified",
            icon=":material/error:",
        )
    st.markdown(ui.escape_md(narrative.text))
    source = f"LLM ({narrative.model})" if narrative.source == "llm" else "deterministic template"
    st.caption(
        f"Written by: {source}" + (f" · LLM error: {narrative.error}" if narrative.error else "")
    )
    with st.expander("Facts sheet (the source of truth for every number)", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [{"fact": f.fact_id, "what": f.label, "value": f.display} for f in narrative.facts]
            ),
            hide_index=True,
            alt="Facts sheet",
        )

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
