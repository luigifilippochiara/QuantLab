"""Run history page: list past runs, inspect a manifest, replay it without the LLM, trace it."""

import labels
import pandas as pd
import streamlit as st

from quantlab.lineage import NEVER_EXECUTED, list_runs, load_run, replay, trace

import ui

settings = ui.get_settings()
catalog = ui.get_catalog(settings)
st.title("Run history")
st.caption(
    "Every run is saved as runs/<run_id>/manifest.json with its outputs. Replay re-executes "
    "the final plan with the recorded approvals and no LLM, then compares every output hash."
)

runs = list_runs(settings.runs_dir)
if not runs:
    st.info("No runs yet. Execute a plan on the Analysis page.", icon=":material/history:")
    st.stop()

st.dataframe(
    pd.DataFrame(
        [
            {
                "run id": r["run_id"],
                "created (UTC)": r["created_at"],
                "status": r["status"],
                "mode": r["mode"],
                "steps run": r["n_steps"],
                "request": r["request"][:120],
            }
            for r in runs
        ]
    ),
    hide_index=True,
    alt="Saved runs",
)

run_id = st.selectbox(
    "Run",
    [r["run_id"] for r in runs],
    format_func=lambda rid: next(f"{rid} · {r['status']}" for r in runs if r["run_id"] == rid),
)
record = load_run(run_id, settings.runs_dir)

with st.container(border=True):
    st.markdown(f"**{record.run_id}** · {ui.STATUS_BADGE.get(record.status, record.status)}")
    st.markdown(f"Request: {ui.escape_md(record.request)}")
    if record.llm:
        st.caption(
            f"Planner: {record.llm.provider} / {record.llm.model} · mode {record.llm.mode} · "
            f"prompt {record.llm.prompt_version} · from cache: {record.llm.from_cache}"
        )
    if record.datasets:
        labels = ", ".join(ui.dataset_label(catalog, d.dataset_id) for d in record.datasets)
        st.caption(f"Datasets: {ui.escape_md(labels)}")
    if record.error:
        st.caption(f"Stopped because: {ui.escape_md(record.error)}")
    st.caption(
        f"Registry version {record.registry_version[:12]} · "
        f"approvals {len(record.approvals)} · edits {len(record.edits)} · "
        f"warnings {len(record.warnings)}"
    )
    if record.param_diff:
        st.markdown("**User edits** (planner-proposed value → value that ran)")
        # Step id -> capability, so each edit is labelled as the plan shows it.
        steps = (record.plan_final or {}).get("steps", [])
        capabilities = {s["id"]: s["capability"] for s in steps}

        def shown(change, value):  # a value as the plan editor labels it
            return labels.param_value_label(capabilities.get(change.step_id), change.param, value)

        st.dataframe(
            pd.DataFrame(
                [
                    {"step": c.step_id,
                     "parameter": labels.param_label(capabilities.get(c.step_id), c.param),
                     "planner proposed": shown(c, c.original), "ran with": shown(c, c.final),
                     "edited at (UTC)": c.edited_at}
                    for c in record.param_diff
                ]
            ),
            hide_index=True,
            alt="User edits",
        )  # fmt: skip
        note = labels.periods_caption(
            (capabilities.get(c.step_id), c.param) for c in record.param_diff
        )
        if note:
            st.caption(note)
    else:
        st.caption("No user edits: every parameter is as the planner proposed it.")
    with st.expander("Manifest (JSON)", icon=":material/data_object:"):
        st.json(record.model_dump(mode="json"), expanded=False)

st.subheader("Replay")
if record.plan_final is None:
    st.caption("This run has no plan, so there is nothing to replay.")
elif record.status in NEVER_EXECUTED:
    st.caption(f"This plan was never executed ({record.status}), so there is nothing to replay.")
elif st.button("Replay without the LLM", type="primary", icon=":material/replay:"):
    ui.render_replay(replay(run_id, settings), catalog)

if record.steps:
    st.subheader("Trace")
    executed = [step.step_id for step in record.steps if step.output_hash]
    step_id = st.selectbox("Step to trace", executed, index=len(executed) - 1)
    ui.render_trace(trace(record, step_id), catalog)
