"""Capability functions: the pre-written, tested quantitative building blocks.

Role in the architecture: these 8 functions are the ONLY code that calls
pandas / numpy for analysis. A plan can only *name* one of them (via the
registry in registry.py); it can never supply code. Each function:

- receives its inputs as Artifacts (value + metadata) and its parameters as
  plain keyword arguments, e.g. `compute_returns(series, method="simple", periods=1)`;
- returns a NEW Artifact whose metadata is set explicitly (kind, frequency,
  return method...), so later steps and the validator know what it is;
- is deterministic: the same inputs always give the same output.

Conventions:
- SERIES and FLAGS artifacts never contain NaN. Warm-up periods (e.g. the first
  window-1 points of a rolling statistic) are dropped, so `n_obs` is always the
  number of usable values and joins are plain inner joins.
- Observation-count rules (min_obs) are enforced by the executor before a
  function is called; functions only raise ValueError for inputs that make the
  computation meaningless (e.g. non-positive prices for returns).
"""

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from quantlab.data import Catalog, load_dataset
from quantlab.types import (
    ANNUALISATION_FACTORS,
    Artifact,
    ArtifactMeta,
    DataType,
    Frequency,
    SeriesKind,
    date_span,
)

# pandas resampling rules. "W-FRI": weeks end on Friday, the last trading day of a
# normal week, so a weekly "last" value is the Friday close. "ME": calendar month
# end (the pandas >= 2.2 alias; plain "M" is deprecated).
RESAMPLE_RULES: dict[str, str] = {"W": "W-FRI", "M": "ME"}

# Lower number = more frequent. Used to refuse "resampling" to a higher frequency.
FREQUENCY_ORDER: dict[Frequency, int] = {Frequency.D: 0, Frequency.W: 1, Frequency.M: 2}


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def derive_meta(base: ArtifactMeta, value: pd.Series | None, **changes: Any) -> ArtifactMeta:
    """Copy an input's metadata for a new artifact, refreshing the sample description.

    Args:
        base: metadata of the (first) input; inherited unless overridden.
        value: the new series, used to set n_obs/start/end (None keeps base's values).
        **changes: fields to override, e.g. kind=SeriesKind.RETURNS.

    Returns:
        A new ArtifactMeta. Notes are NOT inherited: each artifact lists only the
        assumptions made by the step that produced it (the trace shows the chain).
    """
    span: dict[str, Any] = {}
    if value is not None:
        n_obs, start, end = date_span(value)
        span = {"n_obs": n_obs, "start": start, "end": end}
    return replace(base, **{"notes": [], **span, **changes})


def inner_join(left: pd.Series, right: pd.Series, names: tuple[str, str]) -> pd.DataFrame:
    """Align two time series on the dates they share.

    Args:
        left: first series.
        right: second series.
        names: column names for the result, e.g. ("value", "flag").

    Returns:
        A DataFrame with the two columns, only on dates present in both.
    """
    joined = pd.concat({names[0]: left, names[1]: right}, axis=1, join="inner")
    return joined.dropna()


def count_episodes(flags: pd.Series) -> int:
    """Count distinct episodes: contiguous runs of True in a boolean series.

    Args:
        flags: boolean series in time order.

    Returns:
        Number of runs of consecutive True values.
    """
    # An episode starts where a flag is True and the previous one was not.
    starts = flags & ~flags.shift(1, fill_value=False)
    return int(starts.sum())


def merged_sources(*artifacts: Artifact) -> list[str]:
    """Source dataset ids of several artifacts, each id once, in order of first appearance."""
    return list(dict.fromkeys(d for a in artifacts for d in a.meta.source_dataset_ids))


# ---------------------------------------------------------------------------
# 1. load_series
# ---------------------------------------------------------------------------


def load_series(dataset_id: str, catalog: Catalog) -> Artifact:
    """Load a catalog dataset as a SERIES artifact (hash-verified, missing values dropped).

    Args:
        dataset_id: id of a dataset in the catalog.
        catalog: the dataset catalog (passed in by the executor).

    Returns:
        SERIES artifact with kind and frequency taken from the catalog.
    """
    loaded = load_dataset(catalog, dataset_id)
    n_obs, start, end = date_span(loaded.series)
    meta = ArtifactMeta(
        kind=loaded.entry.kind,
        frequency=loaded.entry.frequency,
        source_dataset_ids=[dataset_id],
        n_obs=n_obs,
        start=start,
        end=end,
        n_dropped_missing=loaded.n_missing,
        notes=[
            f"dropped {loaded.n_missing} missing values out of {loaded.n_rows_raw} rows "
            "(not filled: filling would invent prices on holidays)"
        ],
    )
    return Artifact(value=loaded.series, data_type=DataType.SERIES, meta=meta)


# ---------------------------------------------------------------------------
# 2. resample
# ---------------------------------------------------------------------------


def resample(series: Artifact, freq: str, how: str = "last") -> Artifact:
    """Convert a series to a lower frequency (weekly or monthly).

    Args:
        series: SERIES artifact (a level-type series).
        freq: "W" (week ending Friday) or "M" (month end).
        how: "last" (last observation in the period, e.g. Friday close) or "mean".

    Returns:
        SERIES artifact with the same kind and the new frequency.

    Raises:
        ValueError: if asked to go to a HIGHER frequency (that would invent data).
    """
    target = Frequency(freq)
    current = series.meta.frequency
    if current is not None and FREQUENCY_ORDER[target] < FREQUENCY_ORDER[current]:
        raise ValueError(f"cannot resample {current} data to the higher frequency {target}.")

    grouped = series.value.resample(RESAMPLE_RULES[freq])
    # "last" = the value at the end of the period (what you could have traded on);
    # "mean" = the average over the period (smoother, but not an observable price).
    result = grouped.last() if how == "last" else grouped.mean()
    # Periods with no observation at all (e.g. a holiday-only week) come out as NaN: drop them.
    result = result.dropna()
    # pandas labels each period with its END date. If the data stops mid-period
    # (e.g. on a Tuesday), the last bar would carry a future Friday date and a
    # partial-week value; drop it so every bar is a complete period. Conservative:
    # a final month whose calendar end falls on a weekend is dropped too.
    if not result.empty and result.index[-1] > series.value.index[-1]:
        result = result.iloc[:-1]
    meta = derive_meta(
        series.meta,
        result,
        frequency=target,
        notes=[f"resampled to {target} ({RESAMPLE_RULES[freq]}) using the period {how}"],
    )
    return Artifact(value=result, data_type=DataType.SERIES, meta=meta)


# ---------------------------------------------------------------------------
# 3. compute_returns
# ---------------------------------------------------------------------------


def compute_returns(series: Artifact, method: str, periods: int = 1) -> Artifact:
    """Turn a price or index level into returns.

    Args:
        series: SERIES artifact of kind price_level or index_level.
        method: "simple" -> P_t / P_{t-k} - 1, or "log" -> ln(P_t / P_{t-k}).
        periods: k, the number of periods each return spans.

    Returns:
        SERIES artifact of kind returns, recording the return method and periods.

    Raises:
        ValueError: if any level is zero or negative (returns are undefined).
    """
    prices = series.value
    if (prices <= 0).any():
        raise ValueError("returns need strictly positive levels.")
    ratio = prices / prices.shift(periods)
    # Simple returns compound multiplicatively and aggregate across assets;
    # log returns add up over time. The choice changes results, hence approval (G12).
    returns = ratio - 1 if method == "simple" else np.log(ratio)
    # The first `periods` values have no earlier price to compare with.
    returns = returns.dropna()
    meta = derive_meta(
        series.meta,
        returns,
        kind=SeriesKind.RETURNS,
        return_method=method,
        periods=periods,
        notes=[f"{method} returns over {periods} period(s)"],
    )
    return Artifact(value=returns, data_type=DataType.SERIES, meta=meta)


# ---------------------------------------------------------------------------
# 4. difference
# ---------------------------------------------------------------------------


def difference(series: Artifact, periods: int) -> Artifact:
    """Period-on-period change x_t - x_{t-k}, in the series' own units.

    This is the right transform for rates and yields: a move from 1% to 2% is a
    +1 percentage point change, while a "return" of +100% would be meaningless.

    Args:
        series: SERIES artifact (a level).
        periods: k, the lag of the difference.

    Returns:
        SERIES artifact of kind changes.
    """
    changes = (series.value - series.value.shift(periods)).dropna()
    meta = derive_meta(
        series.meta,
        changes,
        kind=SeriesKind.CHANGES,
        periods=periods,
        notes=[f"differences over {periods} period(s), in the input's units"],
    )
    return Artifact(value=changes, data_type=DataType.SERIES, meta=meta)


# ---------------------------------------------------------------------------
# 5. spread
# ---------------------------------------------------------------------------


def spread(a: Artifact, b: Artifact) -> Artifact:
    """Compute a - b on the dates both series share.

    Args:
        a: SERIES artifact (e.g. the 10-year yield).
        b: SERIES artifact with the same frequency (e.g. the 2-year yield).

    Returns:
        SERIES artifact of kind spread.
    """
    joined = inner_join(a.value, b.value, ("a", "b"))
    result = (joined["a"] - joined["b"]).rename("spread")
    meta = derive_meta(
        a.meta,
        result,
        kind=SeriesKind.SPREAD,
        source_dataset_ids=merged_sources(a, b),
        notes=["a - b on the inner join of both date indexes"],
    )
    return Artifact(value=result, data_type=DataType.SERIES, meta=meta)


# ---------------------------------------------------------------------------
# 6. rolling_volatility
# ---------------------------------------------------------------------------


def rolling_volatility(series: Artifact, window: int, annualize: bool = True) -> Artifact:
    """Trailing rolling standard deviation, optionally annualised.

    Args:
        series: SERIES artifact of kind returns, changes or spread.
        window: number of observations in each window.
        annualize: scale to a yearly figure using the input's frequency.

    Returns:
        SERIES artifact of kind volatility. The value at t uses observations
        t-window+1 .. t only (trailing), so it is known at time t.

    Raises:
        ValueError: if annualising and the input has no known frequency.
    """
    # ddof=1: sample standard deviation (divide by n-1), the unbiased variance
    # estimator. With short windows (e.g. 12 weeks) ddof=0 would understate
    # volatility by about 4%.
    rolling_std = series.value.rolling(window=window).std(ddof=1)

    factor = 1.0
    if annualize:
        if series.meta.frequency is None:
            raise ValueError("cannot annualise: the input has no frequency.")
        # Square-root-of-time rule: with independent returns, variance grows
        # linearly with time, so volatility scales with sqrt(periods per year).
        # k-period returns already span k periods, so there are only (factor / k)
        # of them per year.
        factor = ANNUALISATION_FACTORS[series.meta.frequency] / (series.meta.periods or 1)
    volatility = (rolling_std * np.sqrt(factor)).dropna()

    note = (
        f"sample std (ddof=1) over {window} observations, annualised by sqrt({factor:g})"
        if annualize
        else f"sample std (ddof=1) over {window} observations, not annualised"
    )
    meta = derive_meta(series.meta, volatility, kind=SeriesKind.VOLATILITY, notes=[note])
    return Artifact(value=volatility, data_type=DataType.SERIES, meta=meta)


# ---------------------------------------------------------------------------
# 7. flag_regime
# ---------------------------------------------------------------------------


def flag_regime(
    series: Artifact, quantile: float = 0.90, method: str = "expanding", min_periods: int = 52
) -> Artifact:
    """Flag observations above a quantile threshold (e.g. high-volatility weeks).

    Args:
        series: SERIES artifact (typically a volatility or a VIX level).
        quantile: threshold quantile, e.g. 0.90 = above the 90th percentile.
        method: "expanding" (threshold from past data only) or "full_sample"
            (one threshold from the whole sample, which uses future data).
        min_periods: expanding method only, minimum history before flagging starts.

    Returns:
        FLAGS artifact: True where the value is strictly above its threshold.
        Expanding flags start once `min_periods` past values exist.
    """
    x = series.value
    if method == "expanding":
        # The threshold for date t is the quantile of values up to t-1: shift(1)
        # moves each value one step later, so the expanding window ending at t
        # holds x_0..x_{t-1}. The threshold is therefore known BEFORE x_t is
        # observed, the flag could have been computed in real time, and x_t does
        # not move its own threshold. (pandas uses linear interpolation between
        # order statistics, the same default as numpy.quantile.)
        threshold = x.shift(1).expanding(min_periods=min_periods).quantile(quantile)
        note = (
            f"flag = value > expanding {quantile:g} quantile of values up to t-1 "
            f"(at least {min_periods} past values)"
        )
    else:
        # One number computed from ALL observations, including those after t:
        # look-ahead bias. Allowed only with explicit approval (G10).
        threshold = pd.Series(x.quantile(quantile), index=x.index)
        note = f"flag = value > full-sample {quantile:g} quantile (uses future data)"

    has_threshold = threshold.notna()
    flags = (x[has_threshold] > threshold[has_threshold]).astype(bool)
    meta = derive_meta(series.meta, flags, kind=SeriesKind.FLAGS, notes=[note])
    return Artifact(value=flags, data_type=DataType.FLAGS, meta=meta)


# ---------------------------------------------------------------------------
# 8. summary_stats
# ---------------------------------------------------------------------------


def summary_stats(series: Artifact) -> Artifact:
    """Descriptive statistics of a series.

    Args:
        series: SERIES artifact.

    Returns:
        TABLE artifact: count, mean, std, min, max, skew, excess_kurtosis, start, end.
    """
    x = series.value
    n_obs, start, end = date_span(x)
    table = {
        "count": n_obs,
        "mean": float(x.mean()),
        # Sample standard deviation (ddof=1), consistent with rolling_volatility.
        "std": float(x.std(ddof=1)),
        "min": float(x.min()),
        "max": float(x.max()),
        # pandas skew/kurt are the bias-adjusted sample estimators; kurt is EXCESS
        # kurtosis (normal distribution = 0), so fat tails show up as > 0.
        "skew": float(x.skew()),
        "excess_kurtosis": float(x.kurt()),
        "start": start,
        "end": end,
    }
    meta = derive_meta(series.meta, x, kind=None, notes=["std with ddof=1; excess kurtosis"])
    return Artifact(value=table, data_type=DataType.TABLE, meta=meta)
