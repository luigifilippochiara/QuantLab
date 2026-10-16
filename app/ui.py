"""Shared helpers for the Streamlit pages: charts, issue and table rendering, widgets.

Role in the architecture: user interface only. Every number shown comes from the
engine's results (artifacts, facts sheet, run records); nothing here computes
analysis results or talks to an LLM.

Chart colours: one blue for series and one orange for flagged regimes. The pair
was checked with a colour-vision validator against BOTH Streamlit backgrounds,
white and dark (#0e1117): lightness, chroma, colour-blind separation and contrast
all pass, so the same two colours work in either theme without detecting it.
"""

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quantlab.config import Settings, load_settings
from quantlab.data import Catalog, load_catalog
from quantlab.lineage import ReplayReport, TraceResult
from quantlab.narrator import format_number
from quantlab.registry import ParamSpec
from quantlab.scenarios import SCENARIOS_BY_ID
from quantlab.types import Artifact, DataType
from quantlab.validation import Issue, Severity

SERIES_COLOR = "#3987e5"  # blue: the series itself
REGIME_COLOR = "#d95926"  # orange: flagged regimes, and the second measure in grouped bars

SEVERITY_BADGE = {
    Severity.BLOCK: ":red-badge[BLOCK]",
    Severity.APPROVAL: ":orange-badge[APPROVAL]",
    Severity.WARN: ":yellow-badge[WARN]",
}
STATUS_BADGE = {
    "completed": ":green-badge[completed]",
    "ready": ":blue-badge[ready to execute]",
    "planned": ":gray-badge[planned]",
    "blocked": ":red-badge[blocked]",
    "failed": ":red-badge[failed]",
}


# ---------------------------------------------------------------------------
# Settings, catalog, text
# ---------------------------------------------------------------------------


def get_settings() -> Settings:
    """Settings from .env / environment (re-read on every rerun; cheap)."""
    return load_settings()


def get_catalog(settings: Settings) -> Catalog:
    """The dataset catalog."""
    return load_catalog(settings.catalog_path)


def escape_md(text: str) -> str:
    """Escape Markdown so LLM or user text is shown literally (e.g. "$" is not LaTeX)."""
    for char in "\\`*_{}[]<>#+-.!|$~":
        text = text.replace(char, "\\" + char)
    return text


def same_value(a: Any, b: Any) -> bool:
    """Equal value AND type: 12 -> 12.0 counts as a change for the strict validator."""
    return a == b and type(a) is type(b)


def load_scenario(scenario_id: str) -> None:
    """Button callback: put an example workflow's request and datasets into the form."""
    scenario = SCENARIOS_BY_ID[scenario_id]
    st.session_state.request_text = scenario.request
    st.session_state.dataset_select = list(scenario.dataset_ids)
    st.session_state.session = None


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def flag_spans(flags: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Contiguous runs of True as (start, end) dates for shading.

    Each span ends at the next observation's date, so a single flagged week is
    drawn as a band one week wide rather than as an invisible line.
    """
    spans = []
    dates = list(flags.index)
    start = None
    for i, (date, flagged) in enumerate(flags.items()):
        if flagged and start is None:
            start = date
        if not flagged and start is not None:
            spans.append((start, date))
            start = None
        if flagged and i == len(dates) - 1:
            spans.append((start, date))
    return spans


def series_figure(
    series: pd.Series, title: str, name: str, flags: pd.Series | None = None
) -> go.Figure:
    """Line chart of one time series, optionally with flagged periods shaded.

    Args:
        series: the values (DatetimeIndex).
        title: chart title (names the series, so one series needs no legend).
        name: short legend name of the series (shown only when flags are shaded).
        flags: boolean series; True periods are shaded behind the line.

    Returns:
        A plotly Figure.
    """
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            line={"color": SERIES_COLOR, "width": 2},
            name=name,
            hovertemplate="%{x|%Y-%m-%d}: %{y:.4g}<extra></extra>",
        )
    )
    if flags is not None:
        for start, end in flag_spans(flags):
            figure.add_vrect(
                x0=start,
                x1=end,
                fillcolor=REGIME_COLOR,
                opacity=0.22,
                line_width=0,
                layer="below",
            )
        # Legend-only entry so the shading is named, not identified by colour alone.
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"symbol": "square", "size": 12, "color": REGIME_COLOR},
                name="Flagged regime (shaded)",
            )
        )
    figure.update_layout(
        title=title,
        height=320,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        hovermode="x unified",
        showlegend=flags is not None,
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    return figure


def value_label(value: float) -> str:
    """Bar label: whole numbers with thousands separators, other values to one decimal."""
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.1f}"


def bar_figure(labels: list[str], values: list[float], title: str) -> go.Figure:
    """Horizontal bar chart of one measure, each bar labelled with its value.

    The value axis is hidden: the labels carry the numbers, so a second scale
    would only add clutter.

    Args:
        labels: category names (shown top to bottom in the given order).
        values: one value per category.
        title: chart title (names the measure).

    Returns:
        A plotly Figure.
    """
    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={"color": SERIES_COLOR, "cornerradius": 4},
            text=[value_label(v) for v in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x}<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        height=max(220, 32 * len(labels) + 90),
        margin={"l": 10, "r": 40, "t": 50, "b": 10},
        xaxis={"visible": False},
        yaxis={"autorange": "reversed"},
        bargap=0.3,
    )
    return figure


def grouped_bar_figure(
    categories: list[str], groups: dict[str, list[float]], title: str, value_title: str
) -> go.Figure:
    """Vertical grouped bars for up to two measures in the same unit (e.g. mean and p95 ms).

    Args:
        categories: x-axis categories.
        groups: measure name -> one value per category (at most two measures).
        title: chart title.
        value_title: y-axis title (the shared unit).

    Returns:
        A plotly Figure with a legend and value labels.
    """
    figure = go.Figure()
    for (name, values), color in zip(groups.items(), [SERIES_COLOR, REGIME_COLOR], strict=False):
        figure.add_trace(
            go.Bar(
                x=categories,
                y=values,
                name=name,
                marker={"color": color, "cornerradius": 4},
                text=[value_label(v) for v in values],
                textposition="outside",
                cliponaxis=False,
            )
        )
    figure.update_layout(
        title=title,
        barmode="group",
        bargap=0.3,
        bargroupgap=0.08,
        height=320,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        yaxis_title=value_title,
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    return figure


# ---------------------------------------------------------------------------
# Issues, results, lineage
# ---------------------------------------------------------------------------


def render_issue(issue: Issue) -> None:
    """Show one guardrail issue: severity badge, code, step, message and suggestion."""
    where = f"step {issue.step_id}" if issue.step_id else "whole plan"
    st.markdown(f"{SEVERITY_BADGE[issue.severity]} **{issue.code} {issue.name}** · {where}")
    st.markdown(escape_md(issue.message))
    if issue.suggestion:
        st.caption(f"Suggestion: {escape_md(issue.suggestion)}")


def flags_by_input(plan_steps: list[Any], artifacts: dict[str, Artifact]) -> dict[str, pd.Series]:
    """Map a series step id to the flags computed from it (for shading its chart)."""
    shading = {}
    for step in plan_steps:
        if step.capability == "flag_regime" and step.id in artifacts:
            shading[step.inputs.get("series", "")] = artifacts[step.id].value
    return shading


def render_artifact(step_id: str, capability: str, artifact: Artifact, flags: pd.Series | None):
    """Show one step's output: a chart (series), a short count (flags) or a table (TABLE).

    Args:
        step_id: plan step id.
        capability: capability name.
        artifact: the step's output.
        flags: flags to shade on this series' chart, if any.
    """
    meta = artifact.meta
    if artifact.data_type == DataType.SERIES:
        title = f"{step_id} {capability} ({meta.kind}, {meta.frequency})"
        st.plotly_chart(
            series_figure(artifact.value, title, str(meta.kind), flags),
            width="stretch",
            theme="streamlit",
            alt=f"Line chart of {title}",
        )
    elif artifact.data_type == DataType.FLAGS:
        flagged = int(artifact.value.sum())
        st.markdown(
            f"{flagged:,} of {meta.n_obs:,} observations flagged; the flagged periods are "
            "shaded on the chart of the series they were computed from."
        )
    elif artifact.data_type == DataType.TABLE:
        rows = [{"statistic": k, "value": format_number(v)} for k, v in artifact.value.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, alt=f"{capability} table")
    if meta.notes:
        st.caption("Recorded assumptions: " + escape_md("; ".join(meta.notes)))


def describe_params(params: dict[str, Any], edited: dict[str, Any]) -> str:
    """Parameters as text, marking user-edited ones, e.g. "window=60 (edited, was 12)"."""
    return ", ".join(
        f"{name}={value}" + (f" (edited, was {edited[name]})" if name in edited else "")
        for name, value in params.items()
    )


def render_trace(result: TraceResult) -> None:
    """Show the lineage chain of a step and its source files; user-edited params are marked."""
    chain = pd.DataFrame(
        [
            {
                "step": s.step_id,
                "capability": s.capability,
                "inputs": ", ".join(f"{k} <- {v}" for k, v in s.inputs.items()) or "-",
                "params": describe_params(s.params, s.edited) or "-",
                "library call": s.library_calls,
                "rows": s.n_obs,
                "range": f"{s.start} to {s.end}",
                "output sha256": (s.output_hash or "")[:16],
            }
            for s in result.chain
        ]
    )
    st.dataframe(chain, hide_index=True, alt="Lineage chain")
    sources = pd.DataFrame(
        [
            {
                "dataset": s.dataset_id,
                "FRED id": s.fred_id,
                "file sha256": s.sha256,
                "retrieved": s.retrieved_at,
                "range used": f"{s.start} to {s.end}",
                "rows used": s.rows_used,
            }
            for s in result.sources
        ]
    )
    st.dataframe(sources, hide_index=True, alt="Source datasets")


def render_replay(report: ReplayReport) -> None:
    """Show a replay report: overall verdict, dataset checks, per-step hash comparison."""
    if report.all_match:
        st.success(report.message, title="All output hashes match", icon=":material/verified:")
    else:
        st.error(report.message, title="Replay does not match", icon=":material/error:")
    if report.datasets:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "dataset": c.dataset_id,
                        "recorded sha256": c.recorded_sha256[:16],
                        "current sha256": (c.current_sha256 or "missing")[:16],
                        "match": "yes" if c.match else "NO",
                    }
                    for c in report.datasets
                ]
            ),
            hide_index=True,
            alt="Dataset hash checks",
        )
    if report.steps:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "step": s.step_id,
                        "capability": s.capability,
                        "original hash": (s.original_hash or "-")[:16],
                        "replay hash": (s.replay_hash or "-")[:16],
                        "match": "yes" if s.match else "NO",
                    }
                    for s in report.steps
                ]
            ),
            hide_index=True,
            alt="Per-step hash comparison",
        )


# ---------------------------------------------------------------------------
# Parameter widgets constrained by the registry
# ---------------------------------------------------------------------------


def _clamp(value: Any, low: float, high: float, fallback: float) -> float:
    """Keep a widget's starting value inside its allowed range (invalid values -> fallback)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    return min(max(value, low), high)


def param_widget(
    spec_param: ParamSpec, current: Any, key: str, dataset_ids: list[str], disabled: bool = False
) -> Any:
    """Render an input widget for one parameter, limited to what the registry allows.

    Choices become select boxes, numbers become number inputs with the registry's
    min/max, booleans become checkboxes, and dataset ids a select box of catalog ids.

    Args:
        spec_param: the parameter's registry spec.
        current: the value in the plan now (the widget starts from it).
        key: unique widget key.
        dataset_ids: catalog dataset ids (for dataset_id).
        disabled: show the value without allowing changes (e.g. after execution).

    Returns:
        The value chosen in the widget.
    """
    label, help_text = spec_param.name, spec_param.description
    if spec_param.counted_in_periods:
        # The same number means a different span at another frequency (see G14).
        label = f"{spec_param.name} (in periods of the series frequency)"
    if spec_param.name == "dataset_id":
        options = dataset_ids
        index = options.index(current) if current in options else 0
        return st.selectbox(label, options, index=index, key=key, help=help_text, disabled=disabled)
    if spec_param.choices is not None:
        options = list(spec_param.choices)
        index = options.index(current) if current in options else 0
        return st.selectbox(label, options, index=index, key=key, help=help_text, disabled=disabled)
    if spec_param.type == "bool":
        start = current if isinstance(current, bool) else bool(spec_param.default)
        return st.checkbox(label, value=start, key=key, help=help_text, disabled=disabled)
    low, high = spec_param.min, spec_param.max
    if spec_param.type == "int":
        start = int(_clamp(current, low, high, low))
        return st.number_input(
            label, min_value=int(low), max_value=int(high), value=start, step=1, key=key,
            help=help_text, disabled=disabled,
        )  # fmt: skip
    start = float(_clamp(current, low, high, low))
    return st.number_input(
        label, min_value=float(low), max_value=float(high), value=start, step=0.01,
        format="%.2f", key=key, help=help_text, disabled=disabled,
    )  # fmt: skip
