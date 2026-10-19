"""Display labels: the one place that turns codes and names into the words the app shows.

Role in the architecture: user interface only. Plans, run records, the registry,
validation, hashes and the plan cache keep their codes and parameter names (a plan
still says freq="W"); every page shows them through these mappings, so a label is
changed in one place. A name or value without a label is shown as it is, so a new
parameter or code never breaks a page. Guardrail codes (V09...), capability codes
(C07...), identifiers and hashes are deliberately left as they are.
"""

from typing import Any

from quantlab.data import Catalog
from quantlab.registry import capability_code

# Parameter names: (capability, parameter) -> label, or (None, parameter) for a
# parameter that means the same in every capability. `method` means different things
# (a return type, a threshold's window), so it is labelled per capability. Parameters
# without an entry (e.g. resample's `how`, `periods`) keep their names.
PARAM_LABELS: dict[tuple[str | None, str], str] = {
    (None, "dataset_id"): "Dataset",
    (None, "freq"): "Frequency",
    ("compute_returns", "method"): "Return type",
    ("forward_return", "method"): "Return type",
    ("flag_regime", "method"): "Threshold from",
    (None, "quantile"): "Quantile",
    (None, "min_periods"): "Minimum history",
    (None, "direction"): "Flag",
    (None, "window"): "Window",
    (None, "annualize"): "Annualised",
}

# Frequency codes, for resample's `freq` and for every series' metadata.
FREQUENCY_LABELS = {"D": "Daily", "W": "Weekly", "M": "Monthly"}
RETURN_TYPE_LABELS = {"simple": "Simple", "log": "Log"}

# Parameter values: (capability, parameter) -> {code: label}, or (None, parameter).
VALUE_LABELS: dict[tuple[str | None, str], dict[Any, str]] = {
    (None, "freq"): FREQUENCY_LABELS,
    ("compute_returns", "method"): RETURN_TYPE_LABELS,
    ("forward_return", "method"): RETURN_TYPE_LABELS,
    ("flag_regime", "method"): {
        "expanding": "Past data only",
        "full_sample": "Full history (look-ahead)",
    },
    (None, "direction"): {"above": "Above the threshold", "below": "Below the threshold"},
}

# What a series means (types.SeriesKind) and an output's data type (types.DataType).
KIND_LABELS = {
    "price_level": "Price level",
    "index_level": "Index level",
    "rate_level": "Rate level",
    "returns": "Returns",
    "forward_returns": "Forward returns",
    "changes": "Changes",
    "volatility": "Volatility",
    "spread": "Spread",
    "flags": "Flags",
    "other": "Other",
}
DATA_TYPE_LABELS = {"series": "Series", "flags": "Flags", "table": "Table"}

# The statistics of result tables and facts sheets (compare_groups, hac_mean_difference,
# and the facts kept for every series and flag output).
STAT_LABELS = {
    "n_flagged": "Flagged periods",
    "n_normal": "Other periods",
    "n_episodes_flagged": "Flagged episodes",
    "mean_flagged": "Mean (flagged)",
    "mean_normal": "Mean (other)",
    "diff_means": "Difference in means",
    "std_flagged": "Std. dev. (flagged)",
    "std_normal": "Std. dev. (other)",
    "se_hac": "HAC standard error",
    "t_stat": "t-statistic",
    "p_value": "p-value",
    "ci_low": "95% CI (low)",
    "ci_high": "95% CI (high)",
    "n_obs": "Observations",
    "maxlags_used": "Newey-West lags",
    "horizon": "Horizon (periods)",
    "start": "Start",
    "end": "End",
    "last_value": "Latest value",
    "share_flagged": "Share flagged",
    "n_episodes": "Episodes",
}

# Words of a capability name written in capitals, so hac_mean_difference reads
# "HAC mean difference" rather than "Hac mean difference".
TITLE_ACRONYMS = {"hac": "HAC"}


def param_label(capability: str | None, name: str) -> str:
    """A parameter's label, e.g. ("resample", "freq") -> "Frequency"; else its name."""
    return PARAM_LABELS.get((capability, name)) or PARAM_LABELS.get((None, name)) or name


def param_value_label(capability: str | None, name: str, value: Any) -> str:
    """A parameter value's label, e.g. ("resample", "freq", "W") -> "Weekly".

    Booleans read "yes" or "no"; a value without a label is shown as it is.
    """
    if isinstance(value, bool):
        return "yes" if value else "no"
    labels = VALUE_LABELS.get((capability, name)) or VALUE_LABELS.get((None, name)) or {}
    try:
        return labels.get(value, str(value))
    except TypeError:  # an unhashable value (e.g. a list) has no label
        return str(value)


def frequency_label(code: Any) -> str:
    """A frequency code's label, e.g. "W" -> "Weekly"; None -> "unknown"."""
    return "unknown" if code is None else FREQUENCY_LABELS.get(str(code), str(code))


def kind_label(kind: Any) -> str:
    """A series kind's label, e.g. "price_level" -> "Price level"."""
    return "unknown" if kind is None else KIND_LABELS.get(str(kind), str(kind))


def data_type_label(data_type: Any) -> str:
    """A data type's label, e.g. "series" -> "Series"."""
    return "unknown" if data_type is None else DATA_TYPE_LABELS.get(str(data_type), str(data_type))


def stat_label(key: str) -> str:
    """A statistic's label, e.g. "se_hac" -> "HAC standard error"; else the key."""
    return STAT_LABELS.get(key, key)


def capability_title(name: str) -> str:
    """A capability's readable name, e.g. "rolling_volatility" -> "Rolling volatility"."""
    readable = " ".join(TITLE_ACRONYMS.get(word, word) for word in name.split("_"))
    return readable[:1].upper() + readable[1:]


def capability_label(name: str, code: str | None = None) -> str:
    """A capability with its code, e.g. "C06 · Rolling volatility".

    Args:
        name: capability name.
        code: its code as recorded in a run record; when missing it is looked up
            from the registry. A name that is not registered is shown alone.
    """
    code = code or capability_code(name)
    return f"{code} · {capability_title(name)}" if code else capability_title(name)


def dataset_label(catalog: Catalog, dataset_id: str) -> str:
    """A dataset's display name with its id, e.g. "NASDAQ Composite Index (NASDAQCOM)".

    The id stays visible: it is what links the data to its source (FRED). An id that
    is not in the catalog is shown as it is.
    """
    entry = catalog.get(dataset_id)
    return f"{entry.title} ({entry.fred_id})" if entry else dataset_id


def dataset_full_name(catalog: Catalog, dataset_id: str) -> str:
    """A dataset's full name with its id, for hover text, e.g. "Standard & Poor's 500 (SP500)"."""
    entry = catalog.get(dataset_id)
    return f"{entry.full_name} ({entry.fred_id})" if entry else dataset_id


def fact_label(fact_id: str, capability: str | None) -> str:
    """A facts-sheet entry's label, e.g. ("s8.p_value", "hac_mean_difference") ->
    "s8 · HAC mean difference · p-value". The fact id itself is kept elsewhere: it is
    the identifier the grounding check uses."""
    step_id, _, key = fact_id.partition(".")
    parts = [step_id, capability_title(capability) if capability else None, stat_label(key)]
    return " · ".join(part for part in parts if part)
