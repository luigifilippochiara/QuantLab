"""Deterministic execution engine.

Runs a plan step by step using only pre-written Python functions. Nothing in a
plan is ever evaluated as code: a step can only *name* a capability, and that
name is looked up in a fixed allow-list. Same plan + same data = same result.

Every call goes through `run_capability`, which guards it even if the caller
skipped `validate_plan`:
  - the capability must be in the registry and have an implementation here;
  - its parameters must pass the same checks the validator uses.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from core.registry import REGISTRY
from core.validation import Plan, check_params


class CapabilityRefusedError(Exception):
    """Raised when asked to run a capability that is unknown or has invalid parameters."""


# ---------------------------------------------------------------------------
# Capability implementations (one plain function per registered capability)
# ---------------------------------------------------------------------------


def rolling_volatility(prices: pd.Series, window: int, periods_per_year: int) -> pd.Series:
    """Compute annualised rolling volatility from a price series.

    Args:
        prices: prices indexed by date, sorted oldest to newest, all > 0.
        window: number of returns in each rolling window.
        periods_per_year: periods per year used to annualise (252 for daily trading data).

    Returns:
        A Series indexed by date: the annualised volatility of the `window`
        returns ending on each date. Dates without a full window are dropped,
        so the first value is on the date of the (window + 1)-th price.

    Raises:
        TypeError / ValueError: if the input data is unusable.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError(f"prices must be a pandas Series, got {type(prices).__name__}.")
    if prices.isna().any():
        # Refuse rather than guess how missing prices should be filled.
        raise ValueError("prices contain missing values.")
    if (prices <= 0).any():
        # The log of zero or a negative number is undefined.
        raise ValueError("prices must be strictly positive.")
    if len(prices) < window + 1:
        # n prices give n - 1 returns, and one volatility value needs `window` returns.
        raise ValueError(
            f"need at least {window + 1} prices for window={window}, got {len(prices)}."
        )

    # Log returns: ln(P_t / P_{t-1}). Preferred over simple returns for volatility
    # because they add up across periods. The first value is NaN (no previous price).
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # Sample standard deviation over each trailing window (pandas uses ddof=1,
    # i.e. divides by window - 1). The first window - 1 values are NaN.
    rolling_std = log_returns.rolling(window=window).std()

    # Square-root-of-time rule: if returns are independent with constant variance,
    # variance grows linearly with time, so volatility grows with sqrt(time).
    annualised = rolling_std * np.sqrt(periods_per_year)

    # Keep only values computed from a full window, and name the series for display.
    return annualised.dropna().rename(f"rolling_vol_{window}")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

# Allow-list: capability name -> the Python function that implements it.
# A plan can only pick a key from this dict; it has no way to supply code.
CAPABILITY_FUNCTIONS: dict[str, Callable[..., pd.Series]] = {
    "rolling_volatility": rolling_volatility,
}


def run_capability(name: str, data: pd.Series, params: dict) -> pd.Series:
    """Run one capability by name, refusing anything not in the registry.

    Args:
        name: capability name, e.g. "rolling_volatility".
        data: the input data for the capability.
        params: parameter name -> value.

    Returns:
        Whatever the capability produces (for now, a pandas Series).

    Raises:
        CapabilityRefusedError: if the capability is not registered, has no
            implementation, or its parameters are invalid.
    """
    spec = REGISTRY.get(name)
    if spec is None:
        raise CapabilityRefusedError(f"'{name}' is not a registered capability.")

    func = CAPABILITY_FUNCTIONS.get(name)
    if func is None:
        # Registered but not implemented is a bug on our side: refuse, don't guess.
        raise CapabilityRefusedError(f"'{name}' is registered but has no implementation.")

    errors = check_params(spec, params)
    if errors:
        raise CapabilityRefusedError(f"invalid parameters for '{name}': " + " ".join(errors))

    # `**params` unpacks the dict into keyword arguments:
    # {"window": 20, "periods_per_year": 252} -> func(data, window=20, periods_per_year=252)
    return func(data, **params)


def execute_plan(raw_plan: dict, datasets: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """Execute a plan step by step and collect each step's output.

    Call `validate_plan` first to get a full list of problems up front; this
    function still refuses unsafe steps on its own (see `run_capability`).

    Args:
        raw_plan: the plan as a dict.
        datasets: named input data, e.g. {"prices": price_series}.

    Returns:
        A dict mapping each step's output name to the data it produced.

    Raises:
        pydantic.ValidationError: if the plan's structure is wrong.
        CapabilityRefusedError: if a step names an unknown capability or bad params.
        ValueError: if a step's input does not exist.
    """
    plan = Plan.model_validate(raw_plan)

    # The workspace holds every named piece of data: the inputs first, then each
    # step's output. Copied so we never modify the caller's dict.
    workspace = dict(datasets)
    results: dict[str, pd.Series] = {}

    for step in plan.steps:
        if step.input not in workspace:
            raise ValueError(f"step '{step.step_id}': input '{step.input}' does not exist.")

        output = run_capability(step.capability, workspace[step.input], step.params)

        workspace[step.output] = output
        results[step.output] = output

    return results
