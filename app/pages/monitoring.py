"""Monitoring page: activity, quality and governance, guardrails, capabilities, cost, latency.

Every number comes from quantlab.monitoring, computed from all the run records
(one per run, its final state) and the event log (guardrail firings, capability
executions, replays); this page only lays them out.
"""

from typing import Any

import pandas as pd
import streamlit as st

from quantlab.config import LLM_PRICES_PER_MILLION_TOKENS
from quantlab.guardrails import Action
from quantlab.lineage import load_runs
from quantlab.monitoring import (
    COST_STAGES,
    KPI_TARGETS,
    activity,
    cache_savings,
    capability_counts,
    cost_table,
    costed_runs,
    guardrail_counts,
    latency_table,
    narrative_failures,
    planner_refusals,
    read_events,
    recent_failures,
    replay_results,
    review_outcomes,
    target_status,
)

import ui

# The cost chart's height in pixels: the cost table (three rows) plus the cache
# cards under it, so the two columns end level.
COST_CHART_HEIGHT = 310
# Tables: numbers right-aligned under headers that wrap rather than overflow.
NUMBERS_RIGHT = [{"selector": "td, th.col_heading", "props": "text-align: right"}]
# The cost table's columns: whole tokens, and dollars to six decimals so sub-cent
# amounts stay visible.
TOKEN_COLUMNS = [
    "Input tokens (mean)",
    "Input tokens (p95)",
    "Output tokens (mean)",
    "Output tokens (p95)",
]
COST_COLUMNS = ["Cost, US$ (mean)", "Cost, US$ (p95)"]
# A card with a target (monitoring.target_status) sits in a container keyed
# "kpi-met-..." or "kpi-missed-...", whose CSS class these rules tint. Translucent tints,
# like Streamlit's own success and warning alerts, keep the theme's text colour, so the
# text stays readable in light and dark mode. No data keeps the neutral card.
STATUS_CSS = """
<style>
[class*="st-key-kpi-met-"] [data-testid="stMetric"] {
  background-color: rgba(33, 195, 84, 0.14); border-color: rgba(33, 195, 84, 0.45);
}
[class*="st-key-kpi-missed-"] [data-testid="stMetric"] {
  background-color: rgba(255, 164, 33, 0.16); border-color: rgba(255, 164, 33, 0.5);
}
</style>
"""

settings = ui.get_settings()
st.html(STATUS_CSS)  # style only: applied without taking space on the page
st.title("Monitoring")
st.caption(
    f"Computed from the run records in `{settings.runs_dir}` and the event log "
    f"`{settings.events_path.name}`. Cards with a target are green when it is met and "
    "orange when it is not."
)

runs = load_runs(settings.runs_dir)
if not runs:
    st.info(
        "No runs yet. Run something on the Analysis page or the CLI and it appears here.",
        icon=":material/monitoring:",
    )
    st.stop()
events = read_events(settings.events_path)


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


def duration(seconds: float | None) -> str:
    """A duration in seconds, minutes or hours, whichever reads best."""
    if seconds is None:
        return "n/a"
    if seconds < 90:
        return f"{seconds:.1f} s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def number_table(frame: pd.DataFrame, formats: dict[str, str]) -> None:
    """A static table indexed by its first column: each number in its format, right-aligned.

    Args:
        frame: the rows; the first column (e.g. the stage) becomes the row labels.
        formats: column -> format, e.g. "{:,.0f}" (a missing value shows as a dash).
    """
    styled = frame.set_index(frame.columns[0]).style.format(formats, na_rep="—")
    st.table(styled.set_table_styles(NUMBERS_RIGHT))


def card(label: str, value: Any, note: str = "", status: str | None = None, help: str = "") -> None:
    """A KPI card with an optional grey note under the value.

    Args:
        label: the card's title.
        value: the value shown.
        note: a short grey line under the value.
        status: for a card with a target, monitoring.target_status: "met" tints it green;
            "missed" tints it orange and adds "Needs attention" and a warning icon, so
            colour is never the only cue; "no_data" or None keeps the neutral card.
        help: the tooltip next to the label.
    """
    missed = status == "missed"
    metric = {
        "delta": f"Needs attention{' · ' + note if note else ''}" if missed else note or None,
        "delta_color": "orange" if missed else "gray",
        "delta_arrow": "off",
        "icon": ":material/warning:" if missed else None,
        "border": True,
        "help": help or None,
    }
    if status in ("met", "missed"):
        # The key gives the container the CSS class that STATUS_CSS tints.
        with st.container(key=f"kpi-{status}-{label.lower().replace(' ', '-')}"):
            st.metric(label, value, **metric)
    else:
        st.metric(label, value, **metric)


def guardrail_chart(rows: list[dict[str, Any]], title: str, key: str) -> None:
    """Horizontal bars of how often each guardrail fired, coloured by action."""
    st.plotly_chart(
        ui.action_bar_figure(
            [f"{r['code']} {r['name']} · {r['action']}" for r in rows],
            [r["count"] for r in rows],
            [Action(r["action"]) for r in rows],
            title,
        ),
        width="stretch",
        theme="streamlit",
        key=key,
        alt=f"{title}, coloured by action",
    )


# ---------------------------------------------------------------- activity

st.subheader("Activity")
counts = activity(runs)
with st.container(horizontal=True):
    card("Total runs", f"{counts['total']:,}")
    for outcome, label in [("completed", "Completed"), ("blocked", "Blocked"),
                           ("failed", "Failed"), ("rejected", "Rejected by human")]:  # fmt: skip
        card(
            label,
            f"{counts[outcome]:,} ({percent(counts[f'{outcome}_share'])})",
            status=target_status(outcome, counts[outcome]) if outcome in KPI_TARGETS else None,
        )
if counts["awaiting"]:
    st.caption(
        f"{counts['awaiting']} more shown and still awaiting a decision, so the shares add "
        "up to less than 100%."
    )

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
        status=target_status("replay_match_rate", replays["replay_match_rate"]),
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
        status=target_status("n01_failure_rate", narratives["n01_failure_rate"]),
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
fired = guardrail_counts(events, include_unused=False)
if fired:
    guardrail_chart(fired, "Times each guardrail fired", "guardrails-fired")
else:
    st.caption("No guardrail has fired yet.")
with st.expander("Show all guardrails (including unused)"):
    guardrail_chart(guardrail_counts(events), "Times each guardrail fired, all guardrails",
                    "guardrails-all")  # fmt: skip

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
    st.caption("No step has been executed yet.")

# ---------------------------------------------------------------- cost

st.subheader("Cost")
prices = "; ".join(
    f"{model} ${p['input']:.2f} input, ${p['output']:.2f} output per million tokens"
    for model, p in LLM_PRICES_PER_MILLION_TOKENS.items()
)
costed = costed_runs(runs)
table = cost_table(costed, LLM_PRICES_PER_MILLION_TOKENS)
savings = cache_savings(runs, LLM_PRICES_PER_MILLION_TOKENS)
table_col, chart_col = st.columns([3, 2])
with table_col:
    if costed:
        number_table(
            pd.DataFrame(
                [
                    {
                        "Stage": row["stage"],
                        "Input tokens (mean)": row["input_mean"],
                        "Input tokens (p95)": row["input_p95"],
                        "Output tokens (mean)": row["output_mean"],
                        "Output tokens (p95)": row["output_p95"],
                        "Cost, US$ (mean)": row["cost_mean"],
                        "Cost, US$ (p95)": row["cost_p95"],
                    }
                    for row in table
                ]
            ),
            dict.fromkeys(TOKEN_COLUMNS, "{:,.0f}") | dict.fromkeys(COST_COLUMNS, "${:,.6f}"),
        )
    else:
        st.caption("No live or cached runs yet: mock runs call no LLM, so there is no cost.")
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
if costed:
    stage_rows = [row for row in table if row["stage"] in COST_STAGES.values()]
    chart_col.plotly_chart(
        ui.bar_figure(
            [row["stage"] for row in stage_rows],
            [row["cost_total"] or 0.0 for row in stage_rows],
            "Total cost per stage (US$)",
            text=[dollars(row["cost_total"]) for row in stage_rows],
            height=COST_CHART_HEIGHT,
        ),
        width="stretch",
        theme="streamlit",
        alt="Total cost per stage",
    )
    caption = (
        f"Per run, over {len(costed)} live or cached runs; mock runs call no LLM and are left "
        f"out. Total {dollars(table[-1]['cost_total'])}. Prices: {prices}."
    )
    st.caption(caption.replace("$", r"\$"))  # escaped: two $ signs would render as LaTeX

# ---------------------------------------------------------------- latency

st.subheader("Latency")
latency = latency_table(runs)
table_col, chart_col = st.columns([2, 3])
with table_col:
    number_table(
        pd.DataFrame(
            [
                {
                    "Stage": row["stage"],
                    "Runs": row["runs"],
                    "Mean (ms)": row["mean_ms"],
                    "p95 (ms)": row["p95_ms"],
                }
                for row in latency
            ]
        ),
        {"Runs": "{:,}", "Mean (ms)": "{:,.1f}", "p95 (ms)": "{:,.1f}"},
    )
measured = [row for row in latency if row["runs"]]
if measured:
    chart_col.plotly_chart(
        ui.grouped_bar_figure(
            [row["stage"] for row in measured],
            {
                "mean": [row["mean_ms"] for row in measured],
                "95th percentile": [row["p95_ms"] for row in measured],
            },
            "Latency per pipeline stage",
            "milliseconds",
        ),
        width="stretch",
        theme="streamlit",
        alt="Mean and 95th-percentile latency per pipeline stage",
    )
unrecorded = [row["stage"] for row in latency if not row["recorded"]]
st.caption(
    "Each stage over the runs that reached it; approval is the time a person took to "
    f"approve a plan on the Analysis page. Not timed in the run record: {', '.join(unrecorded)}."
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
