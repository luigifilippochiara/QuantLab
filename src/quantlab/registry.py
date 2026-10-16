"""Capability registry: the approved menu of analyses, and the only route to running code.

Role in the architecture:
- The planner shows the LLM `llm_summary()`, generated from this file, so the LLM
  knows exactly which capabilities exist and which parameters they accept.
- The validator checks every plan step against these specs (names, parameter
  types and ranges, input data types and kinds).
- The executor dispatches by looking a step's capability name up in `REGISTRY`
  and calling `spec.func`. A name that is not a key in `REGISTRY` has no
  function attached, so it cannot run: the whitelist holds by construction.

Specs are frozen dataclasses (not Pydantic models) because they are written by
us in code, hold function references, and are never parsed from outside input.
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from quantlab import capabilities
from quantlab.types import Artifact, DataType, SeriesKind

# "date" is an ISO date string, "YYYY-MM-DD" (JSON has no date type).
ParamType = Literal["int", "float", "str", "bool", "date"]

# Special values for CapabilitySpec.output_kind, besides a fixed SeriesKind.
SAME_AS_INPUT = "same_as_input"  # e.g. resample keeps the kind of its input
FROM_CATALOG = "from_catalog"  # load_series takes the kind from the dataset catalog


@dataclass(frozen=True)
class InputSpec:
    """One named input of a capability: a reference to an earlier step's output.

    Fields:
        name: input name used in plans, e.g. "series" or "flags".
        data_types: DataTypes this input accepts.
        description: what the input is for.
        kinds: SeriesKinds accepted (None = any kind).
    """

    name: str
    data_types: tuple[DataType, ...]
    description: str
    kinds: tuple[SeriesKind, ...] | None = None


@dataclass(frozen=True)
class ParamSpec:
    """One parameter of a capability.

    Fields:
        name: parameter name used in plans.
        type: "int", "float", "str", "bool" or "date" (a "YYYY-MM-DD" string).
        description: what it controls, in plain words.
        required: if True the plan must give a value; if False `default` is filled in.
        default: value used when the plan omits an optional parameter.
        min: smallest allowed value (numbers only, inclusive).
        max: largest allowed value (numbers only, inclusive).
        choices: allowed values (strings only).
        counted_in_periods: True for window-type parameters counted in periods of
            the input's frequency (12 means 12 weeks on weekly data, 12 days on
            daily data); used by G14 and to label the edit widget.
        nullable: True if null is a valid value meaning "not set" (the default is
            then null), e.g. filter_period's start and end. Every other
            parameter rejects null.
    """

    name: str
    type: ParamType
    description: str
    required: bool = True
    default: Any = None
    min: float | None = None
    max: float | None = None
    choices: tuple[str, ...] | None = None
    counted_in_periods: bool = False
    nullable: bool = False


@dataclass(frozen=True)
class Assumption:
    """A modelling assumption a capability makes, shown to the user.

    Fields:
        id: short identifier, e.g. "return_method".
        text: the assumption in plain words.
        requires_approval: if True the user must explicitly approve it (guardrail G12).
        param: the parameter whose chosen value the assumption is about, if any.
    """

    id: str
    text: str
    requires_approval: bool = False
    param: str | None = None


@dataclass(frozen=True)
class MinObsRule:
    """How many observations a capability needs (checked at run time by guardrail R02).

    Fields:
        description: the rule in words, shown to users and the LLM.
        min_obs: absolute minimum number of usable (aligned) observations.
        window_param: if set, also require window_multiple x params[window_param].
        window_multiple: multiplier for window_param.
        output_min_obs: if set, the step's OUTPUT must also have this many
            observations (filter_period: the selected period); checked after the step.
    """

    description: str
    min_obs: int = 1
    window_param: str | None = None
    window_multiple: int = 1
    output_min_obs: int | None = None


@dataclass(frozen=True)
class CapabilitySpec:
    """Everything the system knows about one capability.

    Fields:
        name: unique name used in plans.
        description: what it does, written for the LLM and the user.
        func: the wrapper function in capabilities.py (never shown to the LLM).
        inputs: named inputs (references to earlier steps).
        params: parameters.
        output_type: DataType of the output.
        output_kind: a SeriesKind, SAME_AS_INPUT, FROM_CATALOG, or None (tables).
        min_obs: observation-count rule.
        assumptions: modelling assumptions (some need approval).
        library_calls: the underlying library calls, for traceability.
        reads_catalog: True only for load_series; the executor passes the catalog.
        meta_from_params: output metadata fields set from a parameter, e.g.
            {"frequency": "freq"} for resample. The validator's dry run uses this
            to predict output metadata without running anything.
    """

    name: str
    description: str
    func: Callable[..., Artifact]
    inputs: tuple[InputSpec, ...]
    params: tuple[ParamSpec, ...]
    output_type: DataType
    output_kind: SeriesKind | str | None
    min_obs: MinObsRule
    assumptions: tuple[Assumption, ...]
    library_calls: str
    reads_catalog: bool = False
    meta_from_params: dict[str, str] = field(default_factory=dict)

    def param(self, name: str) -> ParamSpec | None:
        """Return the ParamSpec called `name`, or None."""
        return next((p for p in self.params if p.name == name), None)

    def input(self, name: str) -> InputSpec | None:
        """Return the InputSpec called `name`, or None."""
        return next((i for i in self.inputs if i.name == name), None)


# Kinds that are levels (things you can aggregate by "last" or difference).
LEVEL_KINDS = (
    SeriesKind.PRICE_LEVEL,
    SeriesKind.INDEX_LEVEL,
    SeriesKind.RATE_LEVEL,
    SeriesKind.SPREAD,
    SeriesKind.VOLATILITY,
    SeriesKind.OTHER,
)


LOAD_SERIES = CapabilitySpec(
    name="load_series",
    description="Load one dataset from the catalog as a daily time series (hash-verified).",
    func=capabilities.load_series,
    inputs=(),
    params=(
        ParamSpec(
            "dataset_id",
            "str",
            "Id of a dataset from the dataset list (e.g. NASDAQCOM).",
        ),
    ),
    output_type=DataType.SERIES,
    output_kind=FROM_CATALOG,
    min_obs=MinObsRule("at least 1 observation"),
    assumptions=(
        Assumption(
            "snapshot",
            "Uses the committed FRED snapshot, verified by SHA-256; missing values "
            "(holidays) are dropped, not filled.",
        ),
    ),
    library_calls="pandas.read_csv; hashlib.sha256 (file hash check)",
    reads_catalog=True,
)

RESAMPLE = CapabilitySpec(
    name="resample",
    description=(
        "Convert a level series to weekly (week ending Friday) or monthly (month end) "
        "frequency, or keep it daily (D, unchanged). Resample levels BEFORE computing "
        "returns or changes."
    ),
    func=capabilities.resample,
    inputs=(InputSpec("series", (DataType.SERIES,), "Level series to aggregate.", LEVEL_KINDS),),
    params=(
        ParamSpec(
            "freq",
            "str",
            "Target frequency: D (keep the daily business-day series), W (weekly) or M (monthly).",
            choices=("D", "W", "M"),
        ),
        ParamSpec(
            "how",
            "str",
            "last = value at period end (e.g. Friday close); mean = period average.",
            required=False,
            default="last",
            choices=("last", "mean"),
        ),
    ),
    output_type=DataType.SERIES,
    output_kind=SAME_AS_INPUT,
    min_obs=MinObsRule("at least 2 observations", min_obs=2),
    assumptions=(
        Assumption(
            "period_end",
            "Weeks end on Friday (W-FRI) and months at calendar month end (ME); "
            "periods with no data are dropped.",
        ),
    ),
    library_calls="pandas.Series.resample('W-FRI' | 'ME').last() | .mean()",
    meta_from_params={"frequency": "freq"},
)

COMPUTE_RETURNS = CapabilitySpec(
    name="compute_returns",
    description="Returns of a price or index level. Not for rates or yields: use difference.",
    func=capabilities.compute_returns,
    inputs=(
        InputSpec(
            "series",
            (DataType.SERIES,),
            "Price or index level series.",
            (SeriesKind.PRICE_LEVEL, SeriesKind.INDEX_LEVEL),
        ),
    ),
    params=(
        ParamSpec(
            "method",
            "str",
            "simple = P_t/P_{t-k} - 1; log = ln(P_t/P_{t-k}).",
            choices=("simple", "log"),
        ),
        ParamSpec(
            "periods",
            "int",
            "k, the number of periods each return spans.",
            required=False,
            default=1,
            min=1,
            max=52,
        ),
    ),
    output_type=DataType.SERIES,
    output_kind=SeriesKind.RETURNS,
    min_obs=MinObsRule(
        "at least 2 x periods observations", min_obs=2, window_param="periods", window_multiple=2
    ),
    assumptions=(
        Assumption(
            "return_method",
            "The return method is a material modelling choice: simple returns compound "
            "multiplicatively, log returns add up over time, and the two give different "
            "averages.",
            requires_approval=True,
            param="method",
        ),
    ),
    library_calls="P / P.shift(k) - 1 (pandas) | numpy.log(P / P.shift(k))",
    meta_from_params={"return_method": "method", "periods": "periods"},
)

DIFFERENCE = CapabilitySpec(
    name="difference",
    description=(
        "Change over k periods, x_t - x_{t-k}, in the series' own units. "
        "The correct transform for rates, yields and spreads."
    ),
    func=capabilities.difference,
    inputs=(InputSpec("series", (DataType.SERIES,), "Level series.", LEVEL_KINDS),),
    params=(ParamSpec("periods", "int", "k, the lag of the difference.", min=1, max=52),),
    output_type=DataType.SERIES,
    output_kind=SeriesKind.CHANGES,
    min_obs=MinObsRule(
        "at least 2 x periods observations", min_obs=2, window_param="periods", window_multiple=2
    ),
    assumptions=(
        Assumption("units", "Changes are in the input's units (percentage points for yields)."),
    ),
    library_calls="pandas.Series - pandas.Series.shift(k)",
    meta_from_params={"periods": "periods"},
)

SPREAD = CapabilitySpec(
    name="spread",
    description="Difference a - b of two series with the same frequency, on their common dates.",
    func=capabilities.spread,
    inputs=(
        InputSpec(
            "a",
            (DataType.SERIES,),
            "First series (minuend).",
            (SeriesKind.RATE_LEVEL, SeriesKind.RETURNS, SeriesKind.CHANGES, SeriesKind.VOLATILITY),
        ),
        InputSpec(
            "b",
            (DataType.SERIES,),
            "Second series (subtracted).",
            (SeriesKind.RATE_LEVEL, SeriesKind.RETURNS, SeriesKind.CHANGES, SeriesKind.VOLATILITY),
        ),
    ),
    params=(),
    output_type=DataType.SERIES,
    output_kind=SeriesKind.SPREAD,
    min_obs=MinObsRule("at least 2 common observations", min_obs=2),
    assumptions=(Assumption("inner_join", "Only dates present in both series are kept."),),
    library_calls="pandas.concat([a, b], join='inner'); a - b",
)

ROLLING_VOLATILITY = CapabilitySpec(
    name="rolling_volatility",
    description=(
        "Trailing rolling volatility (sample standard deviation) of returns, changes or "
        "a spread, annualised by default."
    ),
    func=capabilities.rolling_volatility,
    inputs=(
        InputSpec(
            "series",
            (DataType.SERIES,),
            "Returns, changes or spread series.",
            (SeriesKind.RETURNS, SeriesKind.CHANGES, SeriesKind.SPREAD),
        ),
    ),
    params=(
        ParamSpec(
            "window",
            "int",
            "Observations per rolling window, in periods of the input frequency "
            "(12 = 12 weeks on weekly data, 12 trading days on daily data).",
            min=4,
            max=260,
            counted_in_periods=True,
        ),
        ParamSpec(
            "annualize",
            "bool",
            "Scale by sqrt(periods per year) of the input frequency.",
            required=False,
            default=True,
        ),
    ),
    output_type=DataType.SERIES,
    output_kind=SeriesKind.VOLATILITY,
    min_obs=MinObsRule(
        "at least 3 x window observations", min_obs=12, window_param="window", window_multiple=3
    ),
    assumptions=(
        Assumption(
            "annualisation",
            "Annualised by sqrt(periods per year): 252 daily, 52 weekly, 12 monthly "
            "(divided by k for k-period returns or changes).",
        ),
    ),
    library_calls="pandas.Series.rolling(window).std(ddof=1) * sqrt(factor)",
)

FLAG_REGIME = CapabilitySpec(
    name="flag_regime",
    description=(
        "Flag observations above a quantile threshold, e.g. periods of high volatility. "
        "Prefer method=expanding: it only uses past data."
    ),
    func=capabilities.flag_regime,
    inputs=(
        InputSpec(
            "series",
            (DataType.SERIES,),
            "Series to threshold, e.g. a rolling volatility or the VIX level.",
        ),
    ),
    params=(
        ParamSpec(
            "quantile",
            "float",
            "Threshold quantile, e.g. 0.90 flags values above the 90th percentile.",
            required=False,
            default=0.90,
            min=0.50,
            max=0.99,
        ),
        ParamSpec(
            "method",
            "str",
            "expanding = threshold from data up to t-1 only; full_sample = one threshold "
            "from all data (look-ahead, needs approval).",
            required=False,
            default="expanding",
            choices=("expanding", "full_sample"),
        ),
        ParamSpec(
            "min_periods",
            "int",
            "Expanding method: past observations required before flagging starts, in "
            "periods of the input frequency.",
            required=False,
            default=52,
            min=20,
            max=520,
            counted_in_periods=True,
        ),
    ),
    output_type=DataType.FLAGS,
    output_kind=SeriesKind.FLAGS,
    min_obs=MinObsRule(
        "at least 2 x min_periods observations",
        min_obs=40,
        window_param="min_periods",
        window_multiple=2,
    ),
    assumptions=(
        Assumption(
            "threshold",
            "A flag means the value is strictly above the threshold; the expanding "
            "threshold at t uses values up to t-1 only.",
        ),
    ),
    library_calls=(
        "pandas.Series.shift(1).expanding(min_periods).quantile(q) | pandas.Series.quantile(q)"
    ),
)

SUMMARY_STATS = CapabilitySpec(
    name="summary_stats",
    description="Descriptive statistics: count, mean, std, min, max, skew, excess kurtosis.",
    func=capabilities.summary_stats,
    inputs=(InputSpec("series", (DataType.SERIES,), "Any series."),),
    params=(),
    output_type=DataType.TABLE,
    output_kind=None,
    min_obs=MinObsRule("at least 4 observations", min_obs=4),
    assumptions=(
        Assumption("estimators", "Sample std (ddof=1); skew and excess kurtosis bias-adjusted."),
    ),
    library_calls="pandas.Series.mean / .std(ddof=1) / .min / .max / .skew / .kurt",
)

COMPARE_GROUPS = CapabilitySpec(
    name="compare_groups",
    description=(
        "Average of a series in flagged versus normal periods (e.g. returns in "
        "high-volatility weeks vs other weeks). Descriptive: no statistical test. "
        "Values and flags must have the same frequency."
    ),
    func=capabilities.compare_groups,
    inputs=(
        InputSpec("values", (DataType.SERIES,), "Series to average, e.g. returns."),
        InputSpec("flags", (DataType.FLAGS,), "Regime flags with the same frequency."),
    ),
    params=(),
    output_type=DataType.TABLE,
    output_kind=None,
    min_obs=MinObsRule("at least 10 common observations", min_obs=10),
    assumptions=(
        Assumption("inner_join", "Only dates present in both inputs are kept."),
        Assumption(
            "descriptive",
            "The difference in means describes the sample; it is not a test of significance.",
        ),
    ),
    library_calls="pandas.concat([values, flags], join='inner'); .mean() per group",
)

FILTER_PERIOD = CapabilitySpec(
    name="filter_period",
    description=(
        "Restrict a series to a time interval: EITHER the last N years (lookback_years, "
        "counted back from the series' last observation) OR start and/or end dates, "
        "never both. Placement matters: filtering BEFORE a rolling or expanding "
        "calculation shortens the history it uses; filtering AFTER keeps the full "
        "history for the calculation and only trims what is reported."
    ),
    func=capabilities.filter_period,
    inputs=(InputSpec("series", (DataType.SERIES,), "Any series to restrict to a period."),),
    params=(
        ParamSpec(
            "lookback_years",
            "int",
            "Keep the last N years up to the series' last observation. Leave null when "
            "giving start or end.",
            required=False,
            default=None,
            min=1,
            max=30,
            nullable=True,
        ),
        ParamSpec(
            "start",
            "date",
            "First date to keep, YYYY-MM-DD, inclusive; null = from the first observation. "
            "Leave null when using lookback_years.",
            required=False,
            default=None,
            nullable=True,
        ),
        ParamSpec(
            "end",
            "date",
            "Last date to keep, YYYY-MM-DD, inclusive; null = up to the last observation. "
            "Leave null when using lookback_years.",
            required=False,
            default=None,
            nullable=True,
        ),
    ),
    output_type=DataType.SERIES,
    output_kind=SAME_AS_INPUT,
    min_obs=MinObsRule(
        "at least 2 observations, before and after filtering", min_obs=2, output_min_obs=2
    ),
    assumptions=(
        Assumption(
            "anchor",
            "lookback_years counts back from the series' last observation, not from today, "
            "so a replay gives the same window; start and end are inclusive.",
        ),
    ),
    library_calls=(
        "last date - pandas.DateOffset(years=N); boolean mask on the pandas DatetimeIndex"
    ),
)

FIND_EXTREMES = CapabilitySpec(
    name="find_extremes",
    description=(
        "When a series reached its maximum and/or minimum, and the value there (e.g. when "
        "an index peaked, or when volatility was highest). Works on any series."
    ),
    func=capabilities.find_extremes,
    inputs=(
        InputSpec("series", (DataType.SERIES,), "Any series: a level, returns, a volatility..."),
    ),
    params=(
        ParamSpec(
            "which",
            "str",
            "max, min or both.",
            required=False,
            default="both",
            choices=("max", "min", "both"),
        ),
    ),
    output_type=DataType.TABLE,
    output_kind=None,
    min_obs=MinObsRule("at least 1 observation"),
    assumptions=(
        Assumption(
            "ties",
            "Missing values are ignored. If the extreme value occurs on several dates, the "
            "earliest is reported and tie_count gives the number of dates.",
        ),
    ),
    library_calls="pandas.Series.dropna().max() / .min(); dates where the value equals it",
)


# The registry itself: capability name -> spec. Built from a list so each key is
# guaranteed to equal the spec's own name. This dict IS the whitelist.
REGISTRY: dict[str, CapabilitySpec] = {
    spec.name: spec
    for spec in [
        LOAD_SERIES,
        RESAMPLE,
        COMPUTE_RETURNS,
        DIFFERENCE,
        SPREAD,
        ROLLING_VOLATILITY,
        FLAG_REGIME,
        SUMMARY_STATS,
        COMPARE_GROUPS,
        FILTER_PERIOD,
        FIND_EXTREMES,
    ]
}


def describe_capability(spec: CapabilitySpec) -> dict[str, Any]:
    """Describe one capability as plain data (no function, no library calls).

    Args:
        spec: the capability.

    Returns:
        A JSON-serialisable dict: name, description, inputs, params, output,
        min_obs rule and assumptions. Optional fields that are unset are omitted.
    """
    params = []
    for p in spec.params:
        described: dict[str, Any] = {"name": p.name, "type": p.type, "required": p.required}
        if not p.required:
            described["default"] = p.default
        if p.min is not None:
            described["min"] = p.min
        if p.max is not None:
            described["max"] = p.max
        if p.choices is not None:
            described["choices"] = list(p.choices)
        if p.counted_in_periods:
            described["unit"] = "periods of the input frequency"
        if p.nullable:
            described["nullable"] = True
        described["description"] = p.description
        params.append(described)

    return {
        "name": spec.name,
        "description": spec.description,
        "inputs": [
            {
                "name": i.name,
                "data_types": [t.value for t in i.data_types],
                "kinds": [k.value for k in i.kinds] if i.kinds else "any",
                "description": i.description,
            }
            for i in spec.inputs
        ],
        "params": params,
        "output": {
            "data_type": spec.output_type.value,
            "kind": str(spec.output_kind) if spec.output_kind else None,
        },
        "min_obs": spec.min_obs.description,
        "assumptions": [
            {"text": a.text, "requires_approval": a.requires_approval} for a in spec.assumptions
        ],
    }


def llm_summary() -> str:
    """Return the LLM-facing description of every registered capability, as JSON text.

    Generated from REGISTRY (never hand-written), so the menu the LLM sees can
    never drift from what the validator and executor actually accept.

    Returns:
        A JSON array, one object per capability, in registry order.
    """
    return json.dumps([describe_capability(spec) for spec in REGISTRY.values()], indent=1)


def registry_version() -> str:
    """Return the SHA-256 of `llm_summary()`; stored in every run record and cache key.

    Returns:
        64-character hex digest. Changes whenever any capability's interface changes.
    """
    return hashlib.sha256(llm_summary().encode("utf-8")).hexdigest()
