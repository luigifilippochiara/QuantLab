"""Monitoring page: activity, quality and governance, guardrails, capabilities, cost, latency.

Every number comes from quantlab.monitoring, computed from the run records (one
per run, its final state) and the event log (guardrail firings, capability
executions, replays); this page only lays them out. The filters select runs by
their creation date, planner model and prompt version, and every section then
describes those runs only.
"""

from typing import Any

import pandas as pd
import streamlit as st

from quantlab.config import LLM_PRICES_PER_MILLION_TOKENS
from quantlab.guardrails import Action
from quantlab.lineage import load_runs
from quantlab.monitoring import (
    COST_STAGES,
    STAGES,
    activity,
    cache_savings,
    capability_counts,
    cost_table,
    costed_runs,
    events_for_runs,
    filter_runs,
    guardrail_counts,
    latency_by_run,
    latency_table,
    narrative_failures,
    needs_attention,
    planner_refusals,
    read_events,
    recent_failures,
    replay_results,
    review_outcomes,
    run_date,
    run_model,
    run_prompt_version,
    runs_per_day,
)

import ui

settings = ui.get_settings()
st.title("Monitoring")
st.caption(
    f"Computed from the run records in `{settings.runs_dir}` and the event log "
    f"`{settings.events_path.name}`. A card turns amber when it needs attention."
)

records = load_runs(settings.runs_dir)
if not records:
    st.info(
        "No runs yet. Run something on the Analysis page or the CLI and it appears here.",
        icon=":material/monitoring:",
    )
    st.stop()


def percent(value: float | None) -> str:
    """A rate as a whole percentage, or n/a without data."""
    return "n/a" if value is None else f"{value:.0%}"


def dollars(value: float | None) -> str:
    """US dollars: six decimals below one dollar (LLM calls cost fractions of a cent)."""
    if value is None:
        return "n/a"
    return f"${value:,.6f}" if 0 < value < 1 else f"${value:,.2f}"


def count(value: float | None) -> str:
    """A token count (or a mean of them) as a whole number with thousands separators."""
    return "n/a" if value is None else f"{value:,.0f}"


def milliseconds(value: float | None) -> str:
    """A latency in milliseconds, to one decimal (validation takes a fraction of one)."""
    return "n/a" if value is None else f"{value:,.1f}"


def duration(seconds: float | None) -> str:
    """A duration in seconds, minutes or hours, whichever reads best."""
    if seconds is None:
        return "n/a"
    if seconds < 90:
        return f"{seconds:.1f} s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def card(label: str, value: Any, note: str = "", attention: bool = False, help: str = "") -> None:
    """A KPI card with an optional grey note under the value; amber when it needs attention."""
    if attention:
        st.metric(
            label,
            value,
            delta=f"Needs attention{' · ' + note if note else ''}",
            delta_color="orange",
            delta_arrow="off",
            icon=":material/warning:",
            border=True,
            help=help or None,
        )
    else:
        st.metric(
            label,
            value,
            delta=note or None,
            delta_color="gray",
            delta_arrow="off",
            border=True,
            help=help or None,
        )


# ---------------------------------------------------------------- filters

dates = [run_date(r) for r in records]
first, last = min(dates), max(dates)
date_col, model_col, prompt_col = st.columns([1.2, 1, 1])
picked = date_col.date_input(
    "Created between (UTC)", value=(first, last), min_value=first, max_value=last
)
start, end = (picked[0], picked[-1]) if picked else (first, last)  # mid-selection: one date
models = model_col.multiselect("Planner model", sorted({run_model(r) for r in records}))
prompts = prompt_col.multiselect(
    "Planner prompt version", sorted({run_prompt_version(r) for r in records})
)
runs = filter_runs(records, start, end, models, prompts)
if not runs:
    st.info("No run matches these filters.", icon=":material/filter_alt_off:")
    st.stop()
events = events_for_runs(read_events(settings.events_path), {r.run_id for r in runs})
st.caption(f"Showing {len(runs)} of {len(records)} runs.")

# ---------------------------------------------------------------- activity

st.subheader("Activity")
counts = activity(runs)
with st.container(horizontal=True):
    card("Runs", counts["total"])
    card("Completed", counts["completed"])
    card("Blocked", counts["blocked"])
    card("Failed", counts["failed"], attention=needs_attention("failed", counts["failed"]))
    card("Rejected by the reviewer", counts["rejected"])
if counts["awaiting"]:
    st.caption(f"{counts['awaiting']} shown and still awaiting a decision.")
per_day = runs_per_day(runs)
st.plotly_chart(
    ui.line_figure([d["date"] for d in per_day], [d["runs"] for d in per_day], "Runs per day",
                   "runs"),
    width="stretch",
    theme="streamlit",
    alt="Runs created per day",
)  # fmt: skip

# ---------------------------------------------------------------- quality and governance

st.subheader("Quality and governance")
review = review_outcomes(runs)
replays = replay_results(events)
narratives = narrative_failures(runs)
refusals = planner_refusals(runs)
with st.container(horizontal=True):
    card(
        "Approval rate",
        percent(review["approval_rate"]),
        f"{review['approved']} approved · {review['rejected']} rejected · "
        f"{review['abandoned']} abandoned",
        help="Share of the plans shown on the Analysis page that the reviewer approved "
        "(executed). Abandoned: shown, but neither approved nor rejected.",
    )
    card(
        "Replay match rate",
        percent(replays["replay_match_rate"]),
        f"{replays['replays']} replays",
        attention=needs_attention("replay_match_rate", replays["replay_match_rate"]),
        help="Share of replays whose every output hash matched the original run.",
    )
    card(
        "Edit rate",
        percent(review["edit_rate"]),
        f"of {review['approved']} approved",
        help="Share of approved plans whose parameters the reviewer changed first.",
    )
    card(
        "Approval time (median)",
        duration(review["approval_median_s"]),
        f"p95 {duration(review['approval_p95_s'])}",
        help="Time from the plan being shown to its approval.",
    )
    card(
        "N01 failure rate",
        percent(narratives["n01_failure_rate"]),
        f"of {narratives['narratives']} narratives",
        attention=needs_attention("n01_failure_rate", narratives["n01_failure_rate"]),
        help="Share of narratives that failed the grounding check (N01 UNGROUNDED_NARRATIVE).",
    )
    card(
        "P01 refusal rate",
        percent(refusals["p01_refusal_rate"]),
        f"of {refusals['planned']} planned",
        help="Share of the requests sent to the planner that it refused (P01 PLANNER_REFUSED).",
    )
st.caption(
    "Approval, edit rate and approval time count plans reviewed on the Analysis page"
    + (
        f"; {review['headless']} headless runs (CLI, notebooks), decided by their caller, "
        "are left out."
        if review["headless"]
        else "."
    )
)

# ---------------------------------------------------------------- guardrails

st.subheader("Guardrails")
guardrails = guardrail_counts(events)
if any(row["count"] for row in guardrails):
    st.plotly_chart(
        ui.action_bar_figure(
            [f"{r['code']} {r['name']} · {r['action']}" for r in guardrails],
            [r["count"] for r in guardrails],
            [Action(r["action"]) for r in guardrails],
            "Times each guardrail fired",
        ),
        width="stretch",
        theme="streamlit",
        alt="Times each guardrail fired, coloured by action",
    )
else:
    st.caption("No guardrail fired in these runs.")

# ---------------------------------------------------------------- capabilities

st.subheader("Capabilities")
capabilities = capability_counts(events)
if any(row["count"] for row in capabilities):
    st.plotly_chart(
        ui.bar_figure(
            [ui.capability_label(r["capability"], r["code"]) for r in capabilities],
            [r["count"] for r in capabilities],
            "Steps executed per capability",
        ),
        width="stretch",
        theme="streamlit",
        alt="Executions per capability",
    )
else:
    st.caption("No step was executed in these runs.")

# ---------------------------------------------------------------- cost

st.subheader("Cost")
prices = "; ".join(
    f"{model} ${p['input']:.2f} input, ${p['output']:.2f} output per million tokens"
    for model, p in LLM_PRICES_PER_MILLION_TOKENS.items()
)
costed = costed_runs(runs)
if costed:
    table = cost_table(costed, LLM_PRICES_PER_MILLION_TOKENS)
    left, right = st.columns([3, 2])
    left.table(
        pd.DataFrame(
            [
                {
                    "Stage": row["stage"],
                    "Input tokens (mean)": count(row["input_mean"]),
                    "Input tokens (p95)": count(row["input_p95"]),
                    "Output tokens (mean)": count(row["output_mean"]),
                    "Output tokens (p95)": count(row["output_p95"]),
                    "Cost (mean)": dollars(row["cost_mean"]),
                    "Cost (p95)": dollars(row["cost_p95"]),
                }
                for row in table
            ]
        ).set_index("Stage")
    )
    stage_rows = [row for row in table if row["stage"] in COST_STAGES.values()]
    right.plotly_chart(
        ui.bar_figure(
            [row["stage"] for row in stage_rows],
            [row["cost_total"] or 0.0 for row in stage_rows],
            "Total cost per stage (US$)",
            text=[dollars(row["cost_total"]) for row in stage_rows],
        ),
        width="stretch",
        theme="streamlit",
        alt="Total cost per stage",
    )
    st.caption(
        f"Per run, over {len(costed)} live or cached runs; mock runs call no LLM and are left "
        f"out. Total {dollars(table[-1]['cost_total'])}. Prices: {prices}."
    )
else:
    st.caption("No live or cached runs here: mock runs call no LLM, so there is no cost.")
savings = cache_savings(runs, LLM_PRICES_PER_MILLION_TOKENS)
with st.container(horizontal=True):
    card(
        "Plan-cache hit rate",
        percent(savings["hit_rate"]),
        f"{savings['cache_hits']} of {savings['planner_calls']} planner calls",
    )
    card(
        "Tokens saved by the cache",
        count(savings["saved_input_tokens"] + savings["saved_output_tokens"]),
        f"{count(savings['saved_input_tokens'])} input · "
        f"{count(savings['saved_output_tokens'])} output",
    )
    card("Saved by the cache", dollars(savings["saved_cost"]))

# ---------------------------------------------------------------- latency

st.subheader("Latency")
left, right = st.columns([2, 3])
left.table(
    pd.DataFrame(
        [
            {
                "Stage": row["stage"],
                "Runs": row["runs"],
                "Mean (ms)": milliseconds(row["mean_ms"]),
                "p95 (ms)": milliseconds(row["p95_ms"]),
            }
            for row in latency_table(runs)
        ]
    ).set_index("Stage")
)
by_run = latency_by_run(runs)
right.plotly_chart(
    ui.stacked_bar_figure(
        [f"{row['created_at'][5:16].replace('T', ' ')} · {row['run_id'][-6:]}" for row in by_run],
        {stage: [row[stage] for row in by_run] for stage in STAGES},
        f"Latency per run, by stage (last {len(by_run)} runs)",
        "milliseconds",
    ),
    width="stretch",
    theme="streamlit",
    alt="Latency of the most recent runs, stacked by stage",
)

# ---------------------------------------------------------------- recent errors

st.subheader("Recent errors")
failures = recent_failures(runs)
if failures:
    st.dataframe(
        pd.DataFrame(failures)
        .fillna({"code": "—"})  # stopped without a guardrail (e.g. a step raised an error)
        .rename(columns={"time": "time (UTC)", "run_id": "run"}),
        hide_index=True,
        alt="The most recent blocked or failed runs",
    )
else:
    st.caption("No blocked or failed runs.")

with st.expander("Latest events (raw)", icon=":material/list:"):
    st.dataframe(pd.DataFrame(events[-50:][::-1]), hide_index=True, alt="Latest events")
