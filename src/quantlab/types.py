"""Shared vocabulary: the kinds of data that flow between plan steps.

Role in the architecture: every step of a plan produces one `Artifact`, a value
plus explicit metadata. The metadata (data type, series kind, frequency, return
method...) is what lets the validator reject nonsense *before* running
(e.g. "returns of a yield", "a spread of daily and weekly series") and lets the
capability functions do the right thing at run time (e.g. annualise volatility
with the right number of periods per year). Metadata travels in the Artifact, never in
pandas `.attrs`, so it is always visible and never silently dropped.
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import pandas as pd


class DataType(StrEnum):
    """The shape of a step's value. Capability inputs declare which ones they accept."""

    SERIES = "series"  # float time series: pd.Series with a DatetimeIndex
    FLAGS = "flags"  # boolean time series: pd.Series of bool with a DatetimeIndex
    TABLE = "table"
    # dict of named scalars (e.g. summary statistics), or of named rows that are
    # dicts of scalars (e.g. find_extremes: {"max": {"date": ..., "value": ...}})


class SeriesKind(StrEnum):
    """What a series *means*. Used to block statistically wrong transformations."""

    PRICE_LEVEL = "price_level"  # a traded price or index level (returns make sense)
    INDEX_LEVEL = "index_level"  # an index that is a level, not an investable price (e.g. VIX)
    RATE_LEVEL = "rate_level"  # an interest rate or yield in percent (use changes, not returns)
    RETURNS = "returns"  # period-on-period returns of a price
    CHANGES = "changes"  # period-on-period differences (the right transform for rates)
    VOLATILITY = "volatility"  # a rolling volatility estimate
    SPREAD = "spread"  # the difference between two series
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
        kind: what the series means (None for tables).
        frequency: observation frequency (inherited by tables for reference).
        source_dataset_ids: catalog datasets this artifact was ultimately computed from.
        n_obs: number of observations (rows for series/flags, the sample size for tables).
        start: first date as ISO string (None if not a time series).
        end: last date as ISO string.
        return_method: "simple" or "log" for returns and anything derived from them.
        periods: the lag used by compute_returns / difference (1 = one-period changes).
        n_dropped_missing: load_series only, missing values dropped from the raw file.
        notes: human-readable assumptions recorded at run time (e.g. annualisation factor).
        period_start: filter_period only, first date of the resolved interval (inclusive).
        period_end: filter_period only, last date of the resolved interval (inclusive).
    """

    kind: SeriesKind | None
    frequency: Frequency | None
    source_dataset_ids: list[str]
    n_obs: int
    start: str | None
    end: str | None
    return_method: str | None = None
    periods: int | None = None
    n_dropped_missing: int | None = None
    notes: list[str] = field(default_factory=list)
    period_start: str | None = None
    period_end: str | None = None


@dataclass
class Artifact:
    """The output of one plan step: a value plus its metadata.

    Fields:
        value: a pd.Series (SERIES, FLAGS) or a dict (TABLE).
        data_type: which of the three DataTypes the value is.
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


# Floats are written with 12 significant digits in the canonical form. That is far
# beyond any meaningful precision of these statistics, yet it absorbs last-bit
# differences between CPU builds and library versions (e.g. in a rolling standard
# deviation), so a replay on another machine still matches while any real change
# in a result does not.
CANONICAL_SIGNIFICANT_DIGITS = 12


def _canonical_scalar(value: Any) -> Any:
    """Round floats to CANONICAL_SIGNIFICANT_DIGITS; leave other scalars unchanged."""
    if isinstance(value, bool) or not isinstance(value, float):
        return value
    if math.isnan(value) or math.isinf(value):
        return value
    return float(f"{value:.{CANONICAL_SIGNIFICANT_DIGITS}g}")


def _canonical_json(value: Any) -> Any:
    """Recursively round the floats inside a (nested) dict or list."""
    if isinstance(value, dict):
        return {key: _canonical_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonical_json(item) for item in value]
    return _canonical_scalar(value)


def canonical_text(artifact: Artifact) -> str:
    """Deterministic text form of an artifact's value: the basis of its hash.

    Series and flags become CSV ("date,value" lines); tables become JSON with
    sorted keys. The same value always gives the same text, on
    any machine, so the same inputs always give the same hash. This text is
    also what lineage saves as the step's output file.

    Args:
        artifact: the artifact to serialise.

    Returns:
        The canonical text.
    """
    if isinstance(artifact.value, pd.Series):
        lines = ["date,value"]
        for date, value in artifact.value.items():
            shown = str(bool(value)) if artifact.data_type == DataType.FLAGS else value
            if not isinstance(shown, str):
                shown = repr(_canonical_scalar(float(shown)))
            lines.append(f"{date:%Y-%m-%d},{shown}")
        return "\n".join(lines) + "\n"
    return json.dumps(_canonical_json(artifact.value), sort_keys=True, indent=2) + "\n"


def artifact_hash(artifact: Artifact) -> str:
    """SHA-256 of an artifact's canonical text.

    Args:
        artifact: the artifact.

    Returns:
        64-character hex digest.
    """
    return hashlib.sha256(canonical_text(artifact).encode("utf-8")).hexdigest()
