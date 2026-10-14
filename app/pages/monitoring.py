"""Monitoring page: KPIs, latency per stage, capability usage, guardrail activity, errors.

Everything is aggregated from the append-only event log runs/events.jsonl.
"""

import pandas as pd
import streamlit as st

from quantlab.monitoring import (
    capability_counts,
    guardrail_counts,
    kpis,
    read_events,
    recent_errors,
    stage_latency,
)

import ui

settings = ui.get_settings()
st.title("Monitoring")
st.caption(f"Aggregated from `{settings.events_path}` (one JSON event per line).")

events = read_events(settings.events_path)
if not events:
    st.info("No events yet. Run something on the Analysis page or the CLI.")
    st.stop()


def percent(value: float | None) -> str:
    """Format a rate for a KPI tile."""
    return "n/a" if value is None else f"{value:.0%}"


summary = kpis(events)
with st.container(horizontal=True):
    st.metric("Runs", summary["n_runs"], border=True)
    st.metric("Completed", percent(summary["success_rate"]), border=True)
    st.metric("Blocked", percent(summary["blocked_rate"]), border=True)
    st.metric("Failed", percent(summary["failed_rate"]), border=True)
    st.metric("Replays matching", percent(summary["replay_match_rate"]), border=True)
    st.metric("Plan-cache hit rate", percent(summary["cache_hit_rate"]), border=True)
st.caption(
    f"Rates are shares of finished runs ({summary['n_in_progress']} still awaiting review); "
    f"{summary['n_replays']} replays; cache hit rate counts live/cached plans only."
)

st.subheader("Latency per stage")
latency = [row for row in stage_latency(events) if row["count"]]
if latency:
    left, right = st.columns([3, 2])
    left.plotly_chart(
        ui.grouped_bar_figure(
            [row["stage"] for row in latency],
            {
                "mean": [row["mean_ms"] for row in latency],
                "95th percentile": [row["p95_ms"] for row in latency],
            },
            "Stage latency",
            "milliseconds",
        ),
        width="stretch",
        theme="streamlit",
        alt="Mean and 95th percentile latency per stage",
    )
    right.dataframe(pd.DataFrame(latency), hide_index=True, alt="Latency table")

left, right = st.columns(2)
with left:
    st.subheader("Capability usage")
    counts = capability_counts(events)
    if counts:
        st.plotly_chart(
            ui.bar_figure(list(counts), list(counts.values()), "Steps executed per capability"),
            width="stretch",
            theme="streamlit",
            alt="Executions per capability",
        )
    else:
        st.caption("No steps executed yet.")
with right:
    st.subheader("Guardrail events")
    rows = guardrail_counts(events)
    if rows:
        st.plotly_chart(
            ui.bar_figure(
                [f"{r['code']} {r['name']} ({r['severity']})" for r in rows],
                [r["count"] for r in rows],
                "Times each guardrail was raised",
            ),
            width="stretch",
            theme="streamlit",
            alt="Guardrail issues raised by code",
        )
    else:
        st.caption("No guardrail has fired yet.")

st.subheader("Recent errors")
errors = recent_errors(events)
if errors:
    st.dataframe(pd.DataFrame(errors), hide_index=True, alt="Recent errors")
else:
    st.caption("No errors.")

with st.expander("Latest events (raw)", icon=":material/list:"):
    st.dataframe(pd.DataFrame(events[-50:][::-1]), hide_index=True, alt="Latest events")
