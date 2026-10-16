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
]:
    st.session_state.setdefault(key, default)

# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Datasets")
    datasets = st.multiselect(
        "Datasets available to the planner",
        catalog.ids(),
        key="dataset_select",
        format_func=lambda d: f"{d} · {catalog.entries[d].title}",
    )
    for dataset_id in datasets:
        entry = catalog.entries[dataset_id]
        st.caption(
            f"**{dataset_id}**: {entry.frequency} · {entry.kind} · "
            f"{entry.start_date} to {entry.end_date} · sha256 `{entry.sha256[:12]}`"
        )

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

    st.header("Demo scenarios")
    for scenario in SCENARIOS:
        st.button(
            scenario.title,
            key=f"scenario-{scenario.id}",
            on_click=ui.load_scenario,
            args=(scenario.id,),
            width="stretch",
        )

# ---------------------------------------------------------------- request

st.title("QuantLab")
st.caption(
    "The LLM proposes a plan made only of approved capabilities; the system validates it, "
    "you review and approve it, and a deterministic engine runs it."
)
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
    st.info(
        "Pick a demo scenario in the sidebar or type a request, then generate a plan.",
        icon=":material/lightbulb:",
    )
    st.stop()

# ---------------------------------------------------------------- plan

st.divider()
st.markdown(f"Run `{session.run_id}` · {ui.STATUS_BADGE.get(session.status, session.status)}")
plan = session.plan
result = session.plan_result

st.subheader("Plan")
if plan is None:
    st.error(result.error if result else "No plan.", title="The planner returned no usable plan")
elif plan.cannot_fulfill:
    st.error(ui.escape_md(plan.refusal_reason or ""), title="The planner declined this request")
else:
    dry_run = session.report.dry_run if session.report else {}
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "step": step.id,
                    "capability": step.capability,
                    "inputs": ", ".join(f"{k} <- {v}" for k, v in step.inputs.items()) or "-",
                    "params": ", ".join(f"{k}={v}" for k, v in step.params.items()) or "-",
                    "output (predicted)": " / ".join(
                        str(dry_run[step.id][k]) for k in ("data_type", "kind", "frequency")
                    )
                    if step.id in dry_run
                    else "unknown",
                    "rationale": step.rationale,
                }
                for step in plan.steps
            ]
        ),
        hide_index=True,
        alt="Plan steps",
    )
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

finished = session.status == "completed"

# ---------------------------------------------------------------- edits

if plan is not None and plan.steps and not plan.cannot_fulfill:
    with st.expander("Edit parameters", icon=":material/tune:", expanded=False):
        if finished:
            st.caption("This run has completed; generate a new plan to change it.")
        with st.form(f"edit-{session.run_id}"):
            chosen: dict[str, dict] = {}
            for step in plan.steps:
                spec = REGISTRY.get(step.capability)
                if spec is None or not spec.params:
                    continue
                st.markdown(f"**{step.id} · {step.capability}**")
                columns = st.columns(len(spec.params))
                chosen[step.id] = {}
                for spec_param, column in zip(spec.params, columns, strict=True):
                    with column:
                        chosen[step.id][spec_param.name] = ui.param_widget(
                            spec_param,
                            step.params.get(spec_param.name),
                            f"edit-{session.run_id}-{step.id}-{spec_param.name}",
                            catalog.ids(),
                        )
            submitted = st.form_submit_button(
                "Apply edits and re-validate", icon=":material/rule:", disabled=finished
            )
        if submitted:
            steps = {step.id: step for step in plan.steps}
            changes = {
                step_id: {
                    name: value
                    for name, value in values.items()
                    if not (
                        steps[step_id].params.get(name) == value
                        and type(steps[step_id].params.get(name)) is type(value)
                    )
                }
                for step_id, values in chosen.items()
            }
            changes = {step_id: params for step_id, params in changes.items() if params}
            if changes:
                session.apply_param_edits(changes)
                # The engine cleared earlier approvals; untick the boxes too, so the
                # user approves the plan as it now stands.
                for key in [k for k in st.session_state if str(k).startswith("approve-")]:
                    del st.session_state[key]
                st.rerun()
            else:
                st.caption("No parameter changed.")
        if session.edits:
            st.caption(
                "Edits so far: "
                + "; ".join(f"{e.step_id}.{e.param}: {e.old} -> {e.new}" for e in session.edits)
            )

# ---------------------------------------------------------------- validation

st.subheader("Validation")
report = session.report
approved_ids = {approval.issue_id for approval in session.approvals}
ticked: list[str] = []
if report is not None:
    if not report.issues:
        st.success("No issues: the plan passed every static guardrail.", icon=":material/check:")
    for severity in (Severity.BLOCK, Severity.APPROVAL, Severity.WARN):
        for issue in [i for i in report.issues if i.severity == severity]:
            with st.container(border=True):
                ui.render_issue(issue)
                if severity == Severity.APPROVAL:
                    is_ticked = st.checkbox(
                        "I approve this",
                        key=f"approve-{session.run_id}-{issue.issue_id}",
                        value=issue.issue_id in approved_ids,
                        disabled=finished,
                    )
                    if is_ticked:
                        ticked.append(issue.issue_id)
    for approval in session.approvals:
        st.caption(
            f"Approved {approval.issue_id} at {approval.approved_at} by {approval.approved_by}"
        )

all_ticked = report is not None and all(
    issue.issue_id in ticked for issue in report.approvals_required
)
can_run = report is not None and not report.is_blocked and all_ticked and not finished
if st.button("Execute", type="primary", icon=":material/play_arrow:", disabled=not can_run):
    session.approve([issue_id for issue_id in ticked if issue_id not in approved_ids])
    with st.spinner("Executing and writing the narrative..."):
        session.execute()
        if session.status == "completed":
            session.narrate()
            session.save_record()
    st.rerun()
if report is not None and report.is_blocked:
    st.caption("Execution is disabled while any BLOCK issue remains.")
elif report is not None and not all_ticked and not finished:
    st.caption("Tick every approval to enable execution.")

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

# ---------------------------------------------------------------- trace and record

if records:
    st.subheader("Trace: where does this number come from?")
    executed = [step.id for step in plan.steps if step.id in artifacts]
    chosen_step = st.selectbox(
        "Step to trace",
        executed,
        index=len(executed) - 1,
        format_func=lambda s: f"{s} · {records[s].capability}",
    )
    ui.render_trace(session.trace(chosen_step))

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
        ui.render_replay(st.session_state.replay_report)
