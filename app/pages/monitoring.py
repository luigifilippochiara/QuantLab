"""Monitoring page: activity, quality and governance, capabilities, guardrails, cost, latency.

Every number comes from quantlab.monitoring, computed from the run records (one per
run, its final state) and the event log (guardrail firings, capability executions,
replays); this page only lays them out. It counts the runs started on the Analysis
page whose plan was decided (quantlab.monitoring.monitored_runs); replays count from
every source.
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
    approval_waits,
    cache_savings,
    capability_counts,
    cost_table,
    costed_runs,
    grounded_narratives,
    guardrail_counts,
    latency_table,
    mode_counts,
    monitored_events,
    monitored_runs,
    planner_refusals,
    read_events,
    recent_failures,
    replay_results,
    review_outcomes,
    target_status,
    unused_guardrails,
)

import ui

# Column ratios of the KPI cards, so every label and caption fits on one line (the
# font is unchanged). Activity: one row, wider for Total runs (its mode breakdown)
# and Failed (its "Needs attention" line). Quality and governance: two rows of three
# equal columns, aligned; its six labels need about 1,260 px in one row, more than
# the page has on a laptop screen.
ACTIVITY_COLUMNS = [1.4, 1, 1, 1.45, 1.05]
QUALITY_COLUMNS = [1, 1, 1]
# The cost and latency sections set a table beside a chart. A small gap under the
# section header moves both columns down; the chart's title sits higher in its column
# (a smaller top margin than the default 50 px), so the chart's top lines up with the
# table's.
HEADER_GAP = "xxsmall"  # 0.25rem, plus the page's usual gap between elements
CHART_TOP_MARGIN = 34
# The cost chart's height in pixels: the cost table (three rows) plus the cache
# cards under it, so the two columns end level.
COST_CHART_HEIGHT = 340
# The latency chart's height in pixels: the latency table (five stages) plus its
# caption, so the two columns start and end level.
LATENCY_CHART_HEIGHT = 270
# Row height of an interactive table (st.dataframe), to size one to show every row.
ROW_PX = 35
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

records = load_runs(settings.runs_dir)
if not records:
    st.info(
        "No runs yet. Run something on the Analysis page or the CLI and it appears here.",
        icon=":material/monitoring:",
    )
    st.stop()
runs = monitored_runs(records)
events = monitored_events(read_events(settings.events_path), runs)
if not runs:
    st.info(
        "No decided runs from the Analysis page yet. This page counts the runs started "
        "there once their plan is approved, rejected or blocked; replays count from every "
        "source.",
        icon=":material/monitoring:",
    )


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


def out_of(count: int, noun: str, plural: str = "", word: str = "of") -> str:
    """A card's grey caption naming its denominator, e.g. "of 20 runs" or "over 1 cache hit".

    Args:
        count: the denominator.
        noun: what is counted, singular (e.g. "plan reviewed").
        plural: its plural, when not the noun plus "s" (e.g. "plans reviewed").
        word: "of" for a share, "over" for a mean or a sum.
    """
    counted = noun if count == 1 else plural or f"{noun}s"
    return f"{word} {count:,} {counted}"


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


# ---------------------------------------------------------------- activity

st.subheader("Activity")
counts = activity(runs)
modes = mode_counts(runs)
total_col, *outcome_cols = st.columns(ACTIVITY_COLUMNS)
with total_col:
    card("Total runs", f"{counts['total']:,}", " · ".join(f"{n:,} {m}" for m, n in modes.items()))
for column, (outcome, label) in zip(outcome_cols, [("completed", "Completed"),
        ("blocked", "Blocked"), ("failed", "Failed"), ("rejected", "Rejected by human")],
        strict=True):  # fmt: skip
    with column:
        card(
            label,
            f"{counts[outcome]:,} ({percent(counts[f'{outcome}_share'])})",
            out_of(counts["total"], "run"),
            status=target_status(f"{outcome}_share", counts[f"{outcome}_share"])
            if f"{outcome}_share" in KPI_TARGETS
            else None,
        )

# ---------------------------------------------------------------- quality and governance

st.subheader("Quality and governance")
review = review_outcomes(runs)
replays = replay_results(events)
narratives = grounded_narratives(runs)
refusals = planner_refusals(runs)
approval_col, replay_col, edit_col = st.columns(QUALITY_COLUMNS)
time_col, grounded_col, refused_col = st.columns(QUALITY_COLUMNS)
with approval_col:
    card(
        "Approval rate",
        percent(review["approval_rate"]),
        out_of(review["approved"] + review["rejected"], "plan reviewed", "plans reviewed"),
        help="Share of the decided plans that the reviewer approved (executed): approved / "
        "(approved + rejected).",
    )
with replay_col:
    card(
        "Replay match rate",
        percent(replays["replay_match_rate"]),
        out_of(replays["replays"], "replay"),
        status=target_status("replay_match_rate", replays["replay_match_rate"]),
        help="Share of replays whose every output hash matched the original run.",
    )
with edit_col:
    card(
        "Edit rate",
        percent(review["edit_rate"]),
        out_of(review["approved"], "plan approved", "plans approved"),
        help="Share of approved plans whose parameters the reviewer changed first.",
    )
with time_col:
    card(
        "Approval time (median)",
        duration(review["approval_median_s"]),
        out_of(len(approval_waits(runs)), "plan approved", "plans approved", "over"),
        help="Median time from the plan being shown to its approval; p95 "
        f"{duration(review['approval_p95_s'])}.",
    )
with grounded_col:
    card(
        "Grounded narrative (N01)",
        percent(narratives["grounded_rate"]),
        out_of(narratives["narratives"], "narrative"),
        status=target_status("grounded_rate", narratives["grounded_rate"]),
        help="Share of the runs with a narrative whose narrative passed the grounding check "
        "(no N01 UNGROUNDED_NARRATIVE).",
    )
with refused_col:
    # The share's denominator is every counted run, the Total runs card.
    card(
        "Planner refused (P01)",
        f"{refusals['refused']:,} ({percent(refusals['refused_share'])})",
        out_of(counts["total"], "run"),
        help="Runs the planner refused (P01 PLANNER_REFUSED), and their share of all runs.",
    )

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

# ---------------------------------------------------------------- guardrails

st.subheader("Guardrails")
fired = guardrail_counts(events, include_unused=False)
if fired:
    st.plotly_chart(
        ui.action_bar_figure(
            [f"{r['code']} {r['name']} · {r['action']}" for r in fired],
            [r["count"] for r in fired],
            [Action(r["action"]) for r in fired],
            "Times each guardrail fired",
        ),
        width="stretch",
        theme="streamlit",
        alt="Times each guardrail fired, coloured by action",
    )
else:
    st.caption("No guardrail has fired yet.")
unused = unused_guardrails(events)
with st.expander("Show unused guardrails"):
    if unused:
        st.dataframe(
            pd.DataFrame(unused).rename(columns=str.capitalize),
            hide_index=True,
            height=ROW_PX * (len(unused) + 1) + 3,  # every row, no scrolling
            alt="Guardrails that have not fired",
        )
    else:
        st.caption("Every guardrail has fired at least once.")

# ---------------------------------------------------------------- cost

st.subheader("Cost")
st.space(HEADER_GAP)
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
            out_of(savings["planner_calls"], "planner call"),
        )
        card(
            "Tokens saved by the cache",
            count(savings["saved_input_tokens"] + savings["saved_output_tokens"]),
            out_of(savings["cache_hits"], "cache hit", word="over"),
            help=f"{count(savings['saved_input_tokens'])} input and "
            f"{count(savings['saved_output_tokens'])} output tokens.",
        )
        card(
            "Saved by the cache",
            dollars(savings["saved_cost"]),
            out_of(savings["cache_hits"], "cache hit", word="over"),
        )
if costed:
    stage_rows = [row for row in table if row["stage"] in COST_STAGES.values()]
    chart_col.plotly_chart(
        ui.bar_figure(
            [row["stage"] for row in stage_rows],
            [row["cost_total"] or 0.0 for row in stage_rows],
            "Total cost per stage (US$)",
            text=[dollars(row["cost_total"]) for row in stage_rows],
            height=COST_CHART_HEIGHT,
            top_margin=CHART_TOP_MARGIN,
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
st.space(HEADER_GAP)
latency = latency_table(runs)
table_col, chart_col = st.columns([2, 3], vertical_alignment="top")
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
    unrecorded = [row["stage"] for row in latency if not row["recorded"]]
    st.caption(
        "Each machine stage over the runs that reached it; the time a person takes to "
        "approve is in the Approval time card. Not timed in the run record: "
        f"{', '.join(unrecorded)}."
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
            height=LATENCY_CHART_HEIGHT,
            top_margin=CHART_TOP_MARGIN,
        ),
        width="stretch",
        theme="streamlit",
        alt="Mean and 95th-percentile latency per pipeline stage",
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
