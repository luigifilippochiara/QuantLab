"""Shared vocabulary: the kinds of data that flow between plan steps.

Role in the architecture: every step of a plan produces one `Artifact`, a value
plus explicit metadata. The metadata (data type, series kind, frequency, return
method, horizon...) is what lets the validator reject nonsense *before* running
(e.g. "returns of a yield", "daily flags against weekly returns") and lets the
capability functions do the right thing at run time (e.g. compound simple
returns but sum log returns). Metadata travels in the Artifact, never in
pandas `.attrs`, so it is always visible and never silently dropped.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import pandas as pd


class DataType(StrEnum):
    """The shape of a step's value. Capability inputs declare which ones they accept."""

    SERIES = "series"  # float time series: pd.Series with a DatetimeIndex
    FLAGS = "flags"  # boolean time series: pd.Series of bool with a DatetimeIndex
    TABLE = "table"  # dict of named scalars, e.g. summary statistics
    TEST_RESULT = "test_result"  # dict with "statistic", "p_value" and "details"


class SeriesKind(StrEnum):
    """What a series *means*. Used to block statistically wrong transformations."""

    PRICE_LEVEL = "price_level"  # a traded price or index level (returns make sense)
    INDEX_LEVEL = "index_level"  # an index that is a level, not an investable price (e.g. VIX)
    RATE_LEVEL = "rate_level"  # an interest rate or yield in percent (use changes, not returns)
    RETURNS = "returns"  # period-on-period returns of a price
    CHANGES = "changes"  # period-on-period differences (the right transform for rates)
    VOLATILITY = "volatility"  # a rolling volatility estimate
    SPREAD = "spread"  # the difference between two series
    FORWARD_RETURN = "forward_return"  # the return over the NEXT h periods (future information)
    FLAGS = "flags"  # boolean regime flags
    OTHER = "other"


class Frequency(StrEnum):
    """Observation frequency of a series."""

    D = "D"  # daily (business days)
    W = "W"  # weekly (week ending Friday)
    M = "M"  # monthly (month end)


# Periods per year used to annualise volatility: ~252 trading days, 52 weeks, 12 months.
ANNUALISATION_FACTORS: dict[Frequency, int] = {Frequency.D: 252, Frequency.W: 52, Frequency.M: 12}


@dataclass
class ArtifactMeta:
    """Metadata carried by every artifact, set explicitly by the capability that made it.

    Fields:
        kind: what the series means (None for tables and test results).
        frequency: observation frequency (inherited by tables/test results for reference).
        source_dataset_ids: catalog datasets this artifact was ultimately computed from.
        n_obs: number of observations (rows for series/flags, the sample size for tests).
        start: first date as ISO string (None if not a time series).
        end: last date as ISO string.
        return_method: "simple" or "log" for returns and anything derived from them.
        horizon: forward-return horizon in periods (forward returns and derived results).
        periods: the lag used by compute_returns / difference (1 = one-period changes).
        notes: human-readable assumptions recorded at run time (e.g. annualisation factor).
    """

    kind: SeriesKind | None
    frequency: Frequency | None
    source_dataset_ids: list[str]
    n_obs: int
    start: str | None
    end: str | None
    return_method: str | None = None
    horizon: int | None = None
    periods: int | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class Artifact:
    """The output of one plan step: a value plus its metadata.

    Fields:
        value: a pd.Series (SERIES, FLAGS) or a dict (TABLE, TEST_RESULT).
        data_type: which of the four DataTypes the value is.
        meta: explicit metadata (see ArtifactMeta).
    """

    value: pd.Series | dict[str, Any]
    data_type: DataType
    meta: ArtifactMeta


def date_span(series: pd.Series) -> tuple[int, str | None, str | None]:
    """Return (number of observations, first date, last date) of a time series.

    Args:
        series: a pd.Series with a DatetimeIndex.

    Returns:
        (n_obs, start, end) with dates as "YYYY-MM-DD" strings, or None when empty.
    """
    if series.empty:
        return 0, None, None
    return len(series), series.index[0].strftime("%Y-%m-%d"), series.index[-1].strftime("%Y-%m-%d")
