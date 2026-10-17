"""Capability functions: the pre-written, tested quantitative building blocks.

Role in the architecture: these functions are the ONLY code that calls
pandas / numpy / statsmodels for analysis. A plan can only *name* one of them
(via the registry in registry.py); it can never supply code. Each function:

- receives its inputs as Artifacts (value + metadata) and its parameters as
  plain keyword arguments, e.g. `compute_returns(series, method="simple", periods=1)`;
- returns a NEW Artifact whose metadata is set explicitly (kind, frequency,
  return method, horizon...), so later steps and the validator know what it is;
- is deterministic: the same inputs always give the same output.

Conventions:
- SERIES and FLAGS artifacts never contain NaN. Warm-up periods (e.g. the first
  window-1 points of a rolling statistic) and forward_return's last h points
  (no future price yet) are dropped, so `n_obs` is always the number of usable
  values and joins are plain inner joins.
- Observation-count rules (min_obs) are enforced by the executor before a
  function is called; functions only raise ValueError for inputs that make the
  computation meaningless (e.g. non-positive prices for returns).
"""

import math
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS

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
        A new ArtifactMeta. Notes and the filter_period interval are NOT inherited:
        each artifact describes only the step that produced it (the trace shows
        the chain).
    """
    span: dict[str, Any] = {}
    if value is not None:
        n_obs, start, end = date_span(value)
        span = {"n_obs": n_obs, "start": start, "end": end}
    step_only = {"notes": [], "period_start": None, "period_end": None}
    return replace(base, **{**step_only, **span, **changes})


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
    """Convert a series to a lower frequency (weekly or monthly), or keep it daily.

    Args:
        series: SERIES artifact (a level-type series).
        freq: "D" (keep daily: pass-through), "W" (week ending Friday) or "M" (month end).
        how: "last" (last observation in the period, e.g. Friday close) or "mean";
            ignored for "D".

    Returns:
        SERIES artifact with the same kind and the new frequency.

    Raises:
        ValueError: if asked to go to a HIGHER frequency (that would invent data).
    """
    target = Frequency(freq)
    current = series.meta.frequency
    if current is not None and FREQUENCY_ORDER[target] < FREQUENCY_ORDER[current]:
        raise ValueError(f"cannot resample {current} data to the higher frequency {target}.")

    if target == Frequency.D:
        # Pass-through: the business-day series is kept exactly as loaded. Not
        # pandas resample("D"), which would insert every calendar day and fill
        # weekends and holidays with NaN. This lets a user switch a weekly plan to
        # daily by changing one parameter.
        meta = derive_meta(
            series.meta, series.value, notes=["kept at daily frequency (no resampling)"]
        )
        return Artifact(value=series.value, data_type=DataType.SERIES, meta=meta)

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


# ---------------------------------------------------------------------------
# 9. compare_groups
# ---------------------------------------------------------------------------


def compare_groups(values: Artifact, flags: Artifact) -> Artifact:
    """Average and spread of a series in flagged versus normal periods (descriptive only).

    Args:
        values: SERIES artifact (e.g. weekly returns, or forward returns as the outcome).
        flags: FLAGS artifact with the same frequency (e.g. high-volatility weeks).

    Returns:
        TABLE artifact: n_flagged, n_normal, n_episodes_flagged (distinct runs of
        consecutive flagged periods), mean_flagged, mean_normal, diff_means
        (flagged minus normal), std_flagged, std_normal, start, end.

    Raises:
        ValueError: if either group is empty (a mean of nothing is meaningless).
    """
    joined = inner_join(values.value, flags.value, ("value", "flag"))
    is_flagged = joined["flag"].astype(bool)
    flagged = joined.loc[is_flagged, "value"]
    normal = joined.loc[~is_flagged, "value"]
    if flagged.empty or normal.empty:
        raise ValueError(
            f"both groups need observations: {len(flagged)} flagged, {len(normal)} normal."
        )
    _, start, end = date_span(joined["value"])
    table = {
        "n_flagged": len(flagged),
        "n_normal": len(normal),
        # Episodes, not weeks, measure how much independent evidence the flagged group
        # holds: volatility clusters, so a run of 6 flagged weeks is one stressed
        # period seen 6 times, not 6 independent observations.
        "n_episodes_flagged": count_episodes(is_flagged),
        "mean_flagged": float(flagged.mean()),
        "mean_normal": float(normal.mean()),
        "diff_means": float(flagged.mean() - normal.mean()),
        # Sample standard deviation (ddof=1), consistent with summary_stats and
        # rolling_volatility; flagged periods are expected to be more dispersed.
        "std_flagged": float(flagged.std(ddof=1)),
        "std_normal": float(normal.std(ddof=1)),
        "start": start,
        "end": end,
    }
    # A difference in means only DESCRIBES the sample: with autocorrelated returns
    # and few flagged episodes it says nothing about statistical significance
    # (hac_mean_difference is the test).
    note = "difference in means on the dates both inputs share; no statistical test"
    meta = derive_meta(
        values.meta,
        joined["value"],
        kind=None,
        source_dataset_ids=merged_sources(values, flags),
        notes=[note],
    )
    return Artifact(value=table, data_type=DataType.TABLE, meta=meta)


# ---------------------------------------------------------------------------
# 10. filter_period
# ---------------------------------------------------------------------------


def filter_period(series: Artifact, start: str | None = None, end: str | None = None) -> Artifact:
    """Restrict a series to a time interval between explicit dates.

    At least one of start and end must be given (the validator enforces it with
    G03 before anything runs; the check here is a second line of defence).

    Args:
        series: SERIES artifact.
        start: first date to keep, "YYYY-MM-DD", inclusive (None = from the first observation).
        end: last date to keep, "YYYY-MM-DD", inclusive (None = up to the last observation).

    Returns:
        SERIES artifact with the same kind and frequency. meta.period_start and
        meta.period_end record the resolved interval (inclusive). It may be
        empty; the executor then stops the run with R02.

    Raises:
        ValueError: if neither start nor end is given, if start is not before
            end, or if the input series is empty.
    """
    x = series.value
    if start is None and end is None:
        raise ValueError("give start and/or end.")
    if start is not None and end is not None and start >= end:
        raise ValueError(f"start {start} must be before end {end}.")
    if x.empty:
        raise ValueError("cannot filter an empty series.")

    first = pd.Timestamp(start) if start is not None else x.index[0]
    last = pd.Timestamp(end) if end is not None else x.index[-1]
    kept = x[(x.index >= first) & (x.index <= last)]
    meta = derive_meta(
        series.meta,
        kept,
        period_start=f"{first:%Y-%m-%d}",
        period_end=f"{last:%Y-%m-%d}",
        notes=[f"kept {first:%Y-%m-%d} to {last:%Y-%m-%d} inclusive (explicit dates)"],
    )
    return Artifact(value=kept, data_type=DataType.SERIES, meta=meta)


# ---------------------------------------------------------------------------
# 11. find_extremes
# ---------------------------------------------------------------------------


def find_extremes(series: Artifact, which: str = "both") -> Artifact:
    """When a series reached its maximum and/or minimum, and the value there.

    Args:
        series: SERIES artifact of any kind (a level, returns, a volatility...).
        which: "max", "min" or "both".

    Returns:
        TABLE artifact with one row per extreme, e.g.
        {"max": {"date": "2021-11-19", "value": 16057.44, "tie_count": 1}}.
        tie_count is the number of dates on which the extreme value occurs
        (1 = unique); on a tie the EARLIEST of those dates is reported.

    Raises:
        ValueError: if the series has no non-missing values.
    """
    # A missing value is not an observation, so it can never be an extreme.
    x = series.value.dropna()
    if x.empty:
        raise ValueError("the series has no observations.")
    table = {}
    for extreme in ("max", "min") if which == "both" else (which,):
        value = x.max() if extreme == "max" else x.min()
        # Every date with exactly the extreme value. Exact equality on purpose: a tie
        # means the same recorded value. The index is in time order, so the first
        # date is the earliest, and the count keeps a tie from being silently hidden.
        dates = x.index[x == value]
        table[extreme] = {
            "date": f"{dates[0]:%Y-%m-%d}",
            "value": float(value),
            "tie_count": len(dates),
        }
    note = "missing values ignored; on a tie the earliest date is reported (see tie_count)"
    meta = derive_meta(series.meta, x, kind=None, notes=[note])
    return Artifact(value=table, data_type=DataType.TABLE, meta=meta)


# ---------------------------------------------------------------------------
# 12. forward_return
# ---------------------------------------------------------------------------


def forward_return(series: Artifact, method: str, horizon: int = 1) -> Artifact:
    """Return over the NEXT `horizon` periods, stamped at the date it starts from.

    Value at t = the return from the close of t to the close of t+h:
    ln(P_{t+h} / P_t) for "log", P_{t+h} / P_t - 1 for "simple" (the same
    semantics as compute_returns, pointed forward).

    Why it starts at the close of t: a flag dated t uses data up to the close of t
    (e.g. the volatility of returns ending at t). The forward return begins exactly
    there, so "flag at t -> what happens next" never reuses the data behind the
    flag. It is future information at t, so it is only ever an OUTCOME (G15).

    Args:
        series: SERIES artifact of kind price_level or index_level.
        method: "simple" or "log".
        horizon: h, the number of periods ahead, in periods of the input's frequency.

    Returns:
        SERIES artifact of kind forward_returns, recording the method and horizon.
        The last h dates have no price h periods later yet: they are dropped,
        never filled.

    Raises:
        ValueError: if any level is zero or negative (returns are undefined).
    """
    prices = series.value
    if (prices <= 0).any():
        raise ValueError("returns need strictly positive levels.")
    # shift(-h) brings the price h periods LATER back onto date t, so the ratio at t
    # is P_{t+h} / P_t. The last h dates have nothing to bring back (NaN).
    ratio = prices.shift(-horizon) / prices
    forward = ratio - 1 if method == "simple" else np.log(ratio)
    # Drop, never fill: any filled value would be an invented future price.
    forward = forward.dropna()
    meta = derive_meta(
        series.meta,
        forward,
        kind=SeriesKind.FORWARD_RETURNS,
        return_method=method,
        horizon=horizon,
        notes=[
            f"{method} return from the close of t to the close of t+{horizon}, stamped at t; "
            f"the last {horizon} observation(s) dropped (no future price yet)"
        ],
    )
    return Artifact(value=forward, data_type=DataType.SERIES, meta=meta)


# ---------------------------------------------------------------------------
# 13. hac_mean_difference
# ---------------------------------------------------------------------------


def newey_west_lags(n_obs: int) -> int:
    """Newey and West's rule of thumb for the HAC lag truncation: floor(4 * (T/100)^(2/9)).

    The lag count grows slowly with the sample: enough lags to capture the
    autocorrelation, few enough that the variance estimate stays stable.

    Args:
        n_obs: T, the number of observations in the regression.

    Returns:
        The number of lags (0 for very small samples).
    """
    return math.floor(4 * (n_obs / 100) ** (2 / 9))


def hac_mean_difference(outcome: Artifact, flags: Artifact, maxlags: int | None = None) -> Artifact:
    """Test whether the mean outcome differs between flagged and normal periods.

    Method: OLS regression outcome_t = a + b * flag_t + e_t, with Newey-West
    (HAC) standard errors.

    Why a regression on a 0/1 flag is a difference in means: least squares picks
    the a and b that best fit each group's values. For normal periods (flag = 0)
    the fitted value is a, so a = mean(normal); for flagged periods it is a + b,
    so b = mean(flagged) - mean(normal). The regression adds a standard error for
    that difference.

    Why HAC: ordinary standard errors assume every week is an independent draw.
    Here weeks are not independent: volatility clusters (flagged weeks come in
    runs, and so does the spread of returns), and h-period forward returns
    overlap (consecutive values share h-1 periods). Ordinary errors then overstate
    the confidence. Newey-West errors allow for autocorrelation up to `maxlags`
    periods, weighted down linearly (Bartlett kernel), and for unequal variances.

    Assumptions of the test: the outcome is stationary (returns, not levels); the
    sample is large enough for the normal approximation (validity is asymptotic);
    the flag is known before the outcome is realised (an expanding threshold, and
    forward returns that start at the close of t). It measures an association,
    not a cause or a forecast.

    Args:
        outcome: SERIES artifact of kind returns or forward_returns.
        flags: FLAGS artifact with the same frequency.
        maxlags: Newey-West lag truncation; None means floor(4 * (T/100)^(2/9)),
            raised to at least h-1 for h-period forward returns.

    Returns:
        TABLE artifact: diff_means (b), mean_normal (a), se_hac, t_stat, p_value
        (two-sided), ci_low, ci_high (95%), n_obs, n_flagged, n_normal,
        n_episodes_flagged, maxlags_used, horizon (forward returns only), start, end.

    Raises:
        ValueError: if either group is empty (no difference to estimate).
    """
    joined = inner_join(outcome.value, flags.value, ("outcome", "flag"))
    is_flagged = joined["flag"].astype(bool)
    n_flagged = int(is_flagged.sum())
    n_normal = len(joined) - n_flagged
    if n_flagged == 0 or n_normal == 0:
        raise ValueError(f"both groups need observations: {n_flagged} flagged, {n_normal} normal.")

    is_forward = outcome.meta.kind == SeriesKind.FORWARD_RETURNS
    horizon = outcome.meta.horizon if is_forward else None
    if maxlags is not None:
        maxlags_used = maxlags
        lag_note = f"maxlags={maxlags} (set in the plan)"
    else:
        rule = newey_west_lags(len(joined))
        maxlags_used = rule
        lag_note = f"maxlags={rule} (rule floor(4*(T/100)^(2/9)) with T={len(joined)})"
        if horizon is not None and horizon - 1 > rule:
            # Overlapping h-period forward returns share h-1 periods with their
            # neighbours, so they are autocorrelated up to lag h-1 even if nothing
            # is predictable. Fewer lags than h-1 would ignore part of that overlap.
            maxlags_used = horizon - 1
            lag_note = f"maxlags={maxlags_used} (h-1 for {horizon}-period forward returns)"

    # Explicit design matrix: a constant column (gives a) and the 0/1 flag (gives b).
    design = pd.DataFrame({"const": 1.0, "flag": is_flagged.astype(float)}, index=joined.index)
    fit = OLS(joined["outcome"], design).fit(
        cov_type="HAC",
        # Bartlett weights (1 - lag / (maxlags + 1)) keep the variance estimate
        # positive; no small-sample correction (statsmodels' default).
        cov_kwds={"maxlags": maxlags_used, "kernel": "bartlett", "use_correction": False},
        # Normal (not Student t) p-values and interval: HAC inference is asymptotic.
        use_t=False,
    )
    ci_low, ci_high = fit.conf_int(alpha=0.05).loc["flag"]
    _, start, end = date_span(joined["outcome"])
    table: dict[str, Any] = {
        "diff_means": float(fit.params["flag"]),
        "mean_normal": float(fit.params["const"]),
        "se_hac": float(fit.bse["flag"]),
        "t_stat": float(fit.tvalues["flag"]),
        "p_value": float(fit.pvalues["flag"]),  # two-sided
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_obs": len(joined),
        "n_flagged": n_flagged,
        "n_normal": n_normal,
        # The real amount of independent evidence: flagged periods cluster.
        "n_episodes_flagged": count_episodes(is_flagged),
        "maxlags_used": maxlags_used,
    }
    if horizon is not None:
        table["horizon"] = horizon
    table["start"] = start
    table["end"] = end
    meta = derive_meta(
        outcome.meta,
        joined["outcome"],
        kind=None,
        source_dataset_ids=merged_sources(outcome, flags),
        notes=[f"OLS outcome ~ const + flag, Newey-West (Bartlett) standard errors, {lag_note}"],
    )
    return Artifact(value=table, data_type=DataType.TABLE, meta=meta)
