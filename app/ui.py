"""Shared helpers for the Streamlit pages: charts, issue and table rendering, widgets.

Role in the architecture: user interface only. Every number shown comes from the
engine's results (artifacts, facts sheet, run records); nothing here computes
analysis results or talks to an LLM.

Chart colours: one blue for series and one orange for flagged regimes. The pair
was checked with a colour-vision validator against BOTH Streamlit backgrounds,
white and dark (#0e1117): lightness, chroma, colour-blind separation and contrast
all pass, so the same two colours work in either theme without detecting it.
"""

import html
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quantlab.config import Settings, load_settings
from quantlab.data import Catalog, load_catalog
from quantlab.guardrails import Action, Category, Issue
from quantlab.lineage import ReplayReport, TraceResult
from quantlab.mock_scenarios import SCENARIOS_BY_ID
from quantlab.narrator import format_number
from quantlab.plan import Plan, Step
from quantlab.registry import CapabilitySpec, ParamSpec, capability_code
from quantlab.types import Artifact, DataType, Frequency

SERIES_COLOR = "#3987e5"  # blue: the series itself
REGIME_COLOR = "#d95926"  # orange: flagged regimes, and the second measure in grouped bars

ACTION_BADGE = {
    Action.BLOCK: ":red-badge[BLOCK]",
    Action.APPROVE: ":orange-badge[APPROVE]",
    Action.WARN: ":yellow-badge[WARN]",
}
# Unit of a parameter counted in periods (e.g. a rolling window), by the frequency
# of the step's input: "window: 12 weeks", or "12 trading days" on daily data.
PERIOD_UNITS = {Frequency.D: "trading days", Frequency.W: "weeks", Frequency.M: "months"}
# Words of a capability name written in capitals in step titles, so
# hac_mean_difference reads "HAC mean difference" rather than "Hac mean difference".
TITLE_ACRONYMS = {"hac": "HAC"}
# Header badges of a plan step: its BLOCK and WARN findings and its edits. Approval
# findings get no badge: they are handled one by one in the Validation section.
STEP_BADGES = {
    Action.BLOCK: ":red-badge[:material/block: Blocked]",
    Action.WARN: ":yellow-badge[:material/info: Warning]",
}
EDITED_BADGE = ":blue-badge[:material/edit: edited]"

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


def dataset_label(catalog: Catalog, dataset_id: str) -> str:
    """The label for a dataset everywhere in the UI: FRED series id, then display name.

    Read from data/catalog.json, e.g. "SP500 — S&P 500 Index". The FRED id is used
    for every dataset, so series without a market ticker (DGS10) look the same.
    An id that is not in the catalog is shown as it is.
    """
    entry = catalog.get(dataset_id)
    return f"{entry.fred_id} — {entry.title}" if entry else dataset_id


def dataset_full_name(catalog: Catalog, dataset_id: str) -> str:
    """Hover text for a dataset: FRED series id, then the catalog's full name."""
    entry = catalog.get(dataset_id)
    return f"{entry.fred_id} — {entry.full_name}" if entry else dataset_id


def dataset_names_help(catalog: Catalog) -> str:
    """Tooltip text listing the full name of every catalog dataset, one per line."""
    return "\n".join(f"- {dataset_full_name(catalog, d)}" for d in catalog.ids())


def dataset_caption(catalog: Catalog, dataset_id: str) -> str:
    """One sidebar line for a dataset (label, then details); hovering shows its full name.

    The tooltip is the HTML `title` attribute, so it appears over the text itself.
    Catalog text is escaped before it goes into the HTML.
    """
    entry = catalog.entries[dataset_id]
    details = (
        f"{entry.frequency} · {entry.kind} · {entry.start_date} to {entry.end_date} · "
        f"sha256 {entry.sha256[:12]}"
    )
    label = html.escape(dataset_label(catalog, dataset_id))
    full_name = html.escape(dataset_full_name(catalog, dataset_id), quote=True)
    return f'<span title="{full_name}"><b>{label}</b>: {html.escape(details)}</span>'


# Header styling: sizes and spacing only. Colours are inherited from the active
# Streamlit theme, so the header reads correctly in both light and dark mode.
# The negative top margin pulls the description up under the title; the bottom
# margin is zero because Streamlit already spaces consecutive elements.
HEADER_CSS = """
<style>
.ql-lead { font-size: 1.2rem; font-weight: 600; line-height: 1.45; margin: -0.6rem 0 0 0; }
</style>
"""


def render_header() -> None:
    """Title, product description and the "how it works" line."""
    st.markdown(HEADER_CSS, unsafe_allow_html=True)
    st.title(":material/query_stats: QuantLab", anchor=False)
    st.markdown(
        '<p class="ql-lead">Ask a market question in plain English and get a validated, '
        "fully traceable quantitative answer in seconds.</p>",
        unsafe_allow_html=True,
    )
    st.caption(
        "**How it works:** The LLM proposes a plan made only of approved capabilities. The "
        "system validates it, you review and approve it, and a deterministic engine runs it."
    )


def step_summary(
    step: Step, spec: CapabilitySpec | None, input_frequency: str | None, catalog: Catalog
) -> str:
    """A step's main parameters as short text, e.g. "window: 12 weeks, annualize: yes".

    The main parameters come from the registry (`summary_params`); without them,
    or for an unknown capability, every parameter is shown. Unset values are left
    out, windows counted in periods get the input's unit, and a dataset id is shown
    as its catalog label.

    Args:
        step: the plan step.
        spec: its capability spec (None if the capability is not registered).
        input_frequency: predicted frequency of the step's input ("D", "W", "M"), if known.
        catalog: the dataset catalog.

    Returns:
        The summary text ("" when the step has no parameters).
    """
    if spec is None:
        names = list(step.params)
    else:
        names = list(spec.summary_params) or [p.name for p in spec.params]
    parts = []
    for name in names:
        value = step.params.get(name)
        if value is None:
            continue
        spec_param = spec.param(name) if spec else None
        counted = spec_param is not None and spec_param.counted_in_periods
        if name == "dataset_id" and isinstance(value, str):
            parts.append(dataset_label(catalog, value))
            continue
        if isinstance(value, bool):
            shown = "yes" if value else "no"
        elif counted and input_frequency in PERIOD_UNITS:
            unit = PERIOD_UNITS[input_frequency]  # e.g. "weeks", "trading days"
            shown = f"{value} {unit.removesuffix('s') if value == 1 else unit}"
        else:
            shown = str(value)
        parts.append(f"{name.replace('_', ' ')}: {shown}")
    return ", ".join(parts)


def capability_label(name: str, code: str | None = None) -> str:
    """A capability as shown in the app, e.g. "C06 · rolling_volatility".

    Args:
        name: capability name.
        code: its code as recorded in a run record; when missing it is looked up
            from the registry. A name that is not registered is shown alone.
    """
    code = code or capability_code(name)
    return f"{code} · {name}" if code else name


def step_title(position: int, capability: str) -> str:
    """A step's name in the plan, e.g. "Step 4 · C06 · Rolling volatility".

    An unregistered capability (e.g. one a planner invented) has no code, so its
    title is just "Step 2 · Arima".
    """
    code = capability_code(capability)
    readable = " ".join(TITLE_ACRONYMS.get(word, word) for word in capability.split("_"))
    readable = readable[:1].upper() + readable[1:]  # "rolling volatility" -> "Rolling volatility"
    return f"Step {position} · {code} · {readable}" if code else f"Step {position} · {readable}"


def step_titles(plan: Plan) -> dict[str, str]:
    """Step id -> step name, so findings can name the step as the plan shows it."""
    return {step.id: step_title(i, step.capability) for i, step in enumerate(plan.steps, 1)}


def step_header(position: int, step: Step, summary: str, actions: set[Action], edited: bool) -> str:
    """The collapsed header of a plan step, e.g. "Step 4 · C06 · Rolling volatility · window: ...".

    Badges follow for the step's BLOCK and WARN findings (most severe first) and
    for user-edited parameters, so neither is hidden when the step is collapsed.
    Plan text is escaped, so a planner value cannot inject Markdown.
    """
    title = step_title(position, step.capability)
    text = escape_md(f"{title} · {summary}" if summary else title)
    badges = [STEP_BADGES[a] for a in (Action.BLOCK, Action.WARN) if a in actions]
    if edited:
        badges.append(EDITED_BADGE)
    return " ".join([text, *badges])


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


def render_issue(issue: Issue, titles: dict[str, str] | None = None) -> None:
    """Show one guardrail issue: action badge, code, step, who enforced it, message, suggestion.

    "Enforced by" and the category are written out (from the GUARDRAILS table),
    so nobody has to decode the code's prefix.

    Args:
        issue: the issue.
        titles: step id -> step name (see step_titles), e.g. "Step 3 · C03 ·
            Compute returns"; the step is then named as in the plan, with its id.
    """
    # Issues without a step concern the whole plan, or (N01) the narrative.
    where = "narrative" if issue.category == Category.OUTPUT_GROUNDING else "whole plan"
    if issue.step_id:
        title = (titles or {}).get(issue.step_id)
        where = f"{escape_md(title)} ({issue.step_id})" if title else f"step {issue.step_id}"
    st.markdown(f"{ACTION_BADGE[issue.action]} **{issue.code} {issue.name}** · {where}")
    st.caption(f"Enforced by: **{issue.enforced_by}** · Category: {issue.category}")
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


def render_artifact(
    step_id: str,
    capability: str,
    artifact: Artifact,
    flags: pd.Series | None,
    catalog: Catalog,
) -> None:
    """Show one step's output: a chart (series), a short count (flags) or a table (TABLE).

    Args:
        step_id: plan step id.
        capability: capability name.
        artifact: the step's output.
        flags: flags to shade on this series' chart, if any.
        catalog: the dataset catalog, to label the source datasets.
    """
    meta = artifact.meta
    if artifact.data_type == DataType.SERIES:
        title = f"{step_id} · {capability_label(capability)} ({meta.kind}, {meta.frequency})"
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
        st.dataframe(table_frame(artifact.value), hide_index=True, alt=f"{capability} table")
    sources = ", ".join(dataset_label(catalog, d) for d in meta.source_dataset_ids)
    st.caption("Source: " + escape_md(sources))
    if meta.notes:
        st.caption("Recorded assumptions: " + escape_md("; ".join(meta.notes)))


def table_frame(table: dict[str, Any]) -> pd.DataFrame:
    """A TABLE artifact as a DataFrame: one row per statistic, formatted as in the facts sheet."""
    return pd.DataFrame([{"statistic": k, "value": format_number(v)} for k, v in table.items()])


def describe_params(params: dict[str, Any], edited: dict[str, Any]) -> str:
    """Parameters as text, marking user-edited ones, e.g. "window=60 (edited, was 12)"."""
    return ", ".join(
        f"{name}={value}" + (f" (edited, was {edited[name]})" if name in edited else "")
        for name, value in params.items()
    )


def render_trace(result: TraceResult, catalog: Catalog) -> None:
    """Show the lineage chain of a step and its source files; user-edited params are marked."""
    chain = pd.DataFrame(
        [
            {
                "step": s.step_id,
                "code": s.capability_code or "-",
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
                "dataset": dataset_label(catalog, s.dataset_id),
                "FRED id": s.fred_id,
                "file sha256": s.sha256,
                "retrieved": s.retrieved_at,
                "range used": f"{s.start} to {s.end}",
                "rows used": s.rows_used,
            }
            for s in result.sources
        ]
    )
    st.dataframe(
        sources,
        hide_index=True,
        alt="Source datasets",
        column_config={"dataset": st.column_config.TextColumn(width="large")},
    )


def render_replay(report: ReplayReport, catalog: Catalog) -> None:
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
                        "dataset": dataset_label(catalog, c.dataset_id),
                        "recorded sha256": c.recorded_sha256[:16],
                        "current sha256": (c.current_sha256 or "missing")[:16],
                        "match": "yes" if c.match else "NO",
                    }
                    for c in report.datasets
                ]
            ),
            hide_index=True,
            alt="Dataset hash checks",
            column_config={"dataset": st.column_config.TextColumn(width="large")},
        )
    if report.steps:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "step": s.step_id,
                        "code": s.capability_code or "-",
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
    spec_param: ParamSpec, current: Any, key: str, catalog: Catalog, disabled: bool = False
) -> Any:
    """Render an input widget for one parameter, limited to what the registry allows.

    Choices become select boxes, numbers become number inputs with the registry's
    min/max, booleans become checkboxes, and dataset ids a select box of catalog
    datasets shown by their full names.

    Args:
        spec_param: the parameter's registry spec.
        current: the value in the plan now (the widget starts from it).
        key: unique widget key.
        catalog: the dataset catalog (for dataset_id: choices and full names).
        disabled: show the value without allowing changes (e.g. after execution).

    Returns:
        The value chosen in the widget.
    """
    label, help_text = spec_param.name, spec_param.description
    if spec_param.counted_in_periods:
        # The same number means a different span at another frequency: 12 is 12
        # weeks on weekly data but 12 trading days on daily data.
        label = f"{spec_param.name} (in periods of the series frequency)"
    if spec_param.name == "dataset_id":
        options = catalog.ids()
        index = options.index(current) if current in options else 0
        # Options show each dataset's label; the tooltip lists every dataset's full name.
        return st.selectbox(
            label,
            options,
            index=index,
            key=key,
            format_func=lambda d: dataset_label(catalog, d),
            help=f"{help_text}\n\n{dataset_names_help(catalog)}",
            disabled=disabled,
        )
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
