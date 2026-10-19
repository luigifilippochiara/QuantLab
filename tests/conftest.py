"""Shared test fixtures: synthetic datasets and isolated settings.

Tests never touch the network or the committed FRED snapshot. Instead they
build a small synthetic data folder (FRED-format CSVs + catalog.json with real
hashes) under pytest's tmp_path, using the same dataset ids as the real catalog
so the canned demo plans run unchanged.
"""

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from quantlab.config import Settings
from quantlab.data import Catalog, file_sha256, load_catalog, read_fred_csv
from quantlab.guardrails import fill_defaults
from quantlab.mock_scenarios import SCENARIOS_BY_ID
from quantlab.plan import Plan

# 30 years of business days: long enough for every scenario and for weekly and
# monthly variations of them (e.g. a 52-week window needs 3 x 52 weeks of returns).
N_DAYS = 7800
START = "1996-01-01"


def _business_days(n: int) -> pd.DatetimeIndex:
    """Return n consecutive business days starting at START."""
    return pd.bdate_range(START, periods=n, name="date")


def synthetic_price(n: int, seed: int, start_level: float = 100.0) -> np.ndarray:
    """Simulate a price with volatility clusters (calm 15% vs stressed 45% a year).

    The regime switches with a small daily probability, so high-volatility
    episodes come in contiguous runs, like real markets. That gives
    flag_regime real high-volatility periods to find.
    """
    rng = np.random.default_rng(seed)
    stressed = np.zeros(n, dtype=bool)
    for t in range(1, n):
        switch = rng.random() < 0.01  # ~1% daily chance of switching regime
        stressed[t] = (not stressed[t - 1]) if switch else stressed[t - 1]
    daily_vol = np.where(stressed, 0.45, 0.15) / np.sqrt(252)
    log_returns = rng.normal(0.0003, daily_vol)
    return start_level * np.exp(np.cumsum(log_returns))


def synthetic_rate(n: int, seed: int, start_level: float) -> np.ndarray:
    """Simulate a yield in percent as a bounded random walk (daily steps of ~5bp)."""
    rng = np.random.default_rng(seed)
    return np.clip(start_level + np.cumsum(rng.normal(0, 0.05, n)), 0.05, 20.0)


def write_fred_csv(path: Path, series_id: str, values: np.ndarray, missing_every: int) -> None:
    """Write values in FRED's CSV layout, leaving every `missing_every`-th value empty.

    Args:
        path: destination file.
        series_id: column header, as FRED uses the series id.
        values: one value per business day from START.
        missing_every: 0 for no missing values, k to blank every k-th row.
    """
    dates = _business_days(len(values))
    lines = [f"observation_date,{series_id}"]
    for i, (day, value) in enumerate(zip(dates, values, strict=True)):
        blank = missing_every and i % missing_every == missing_every - 1
        lines.append(f"{day:%Y-%m-%d}," + ("" if blank else f"{value:.4f}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


SYNTHETIC_SERIES = [
    # (id, kind, title, generator)
    ("SP500", "price_level", "Synthetic S&P 500", lambda: synthetic_price(N_DAYS, 6, 2000.0)),
    ("NASDAQCOM", "price_level", "Synthetic equity index", lambda: synthetic_price(N_DAYS, 1)),
    ("VIXCLS", "index_level", "Synthetic VIX", lambda: 12 + synthetic_price(N_DAYS, 2, 8) / 10),
    ("DGS10", "rate_level", "Synthetic 10-Year yield", lambda: synthetic_rate(N_DAYS, 3, 5.0)),
    ("DGS2", "rate_level", "Synthetic 2-Year yield", lambda: synthetic_rate(N_DAYS, 4, 3.5)),
    ("DEXSZUS", "price_level", "Synthetic CHF per USD", lambda: synthetic_price(N_DAYS, 5, 1.5)),
]
# Quote conventions, as in data/catalog.json (the narrator states them).
QUOTE_CONVENTIONS = {
    "DEXSZUS": "USD/CHF, in Swiss francs per US dollar: a negative return means the franc "
    "strengthened against the dollar, a positive one that it weakened.",
}


def build_synthetic_data_dir(data_dir: Path, missing_every: int = 50) -> Path:
    """Create snapshots/ and catalog.json for the synthetic datasets.

    Args:
        data_dir: folder to create the data in.
        missing_every: blank every k-th value (50 -> 2% missing, below the E03 threshold).

    Returns:
        The path of the written catalog.json.
    """
    snapshots = data_dir / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    entries = []
    for series_id, kind, title, generator in SYNTHETIC_SERIES:
        path = snapshots / f"{series_id}.csv"
        write_fred_csv(path, series_id, generator(), missing_every)
        series, n_rows, n_missing = read_fred_csv(path)
        entries.append(
            {
                "dataset_id": series_id,
                "fred_id": series_id,
                "title": title,
                "full_name": f"{title}, full name",
                "description": f"{title} (synthetic test data).",
                "units": "test units",
                "frequency": "D",
                "kind": kind,
                "source_url": "synthetic",
                "file": f"snapshots/{series_id}.csv",
                "retrieved_at": "2026-01-01T00:00:00Z",
                "sha256": file_sha256(path),
                "n_rows": n_rows,
                "n_missing": n_missing,
                "start_date": f"{series.index[0]:%Y-%m-%d}",
                "end_date": f"{series.index[-1]:%Y-%m-%d}",
                **({"quote_convention": QUOTE_CONVENTIONS[series_id]}
                   if series_id in QUOTE_CONVENTIONS else {}),
            }
        )  # fmt: skip
    catalog_path = data_dir / "catalog.json"
    catalog_path.write_text(json.dumps({"datasets": entries}, indent=2), encoding="utf-8")
    return catalog_path


@pytest.fixture(scope="session")
def synthetic_data_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the synthetic data folder once per test session (it takes ~0.3 s)."""
    data_dir = tmp_path_factory.mktemp("template") / "data"
    build_synthetic_data_dir(data_dir)
    return data_dir


@pytest.fixture
def settings(tmp_path: Path, synthetic_data_template: Path) -> Settings:
    """Settings pointing at a private copy of the synthetic data and an empty runs folder.

    Each test gets its own copy, so a test that tampers with a file cannot affect others.
    """
    data_dir = tmp_path / "data"
    shutil.copytree(synthetic_data_template, data_dir)
    # Provider "anthropic" pairs with FakeAnthropicClient; OpenAI tests switch it
    # with dataclasses.replace(settings, llm_provider="openai").
    return Settings(
        anthropic_api_key=None,
        planner_model="test-model",
        narrator_model="test-model",
        llm_mode="mock",
        llm_provider="anthropic",
        data_dir=data_dir,
        runs_dir=tmp_path / "runs",
    )


@pytest.fixture
def catalog(settings: Settings) -> Catalog:
    """The synthetic catalog, loaded."""
    return load_catalog(settings.catalog_path)


# ---------------------------------------------------------------------------
# Plans used by tests, written exactly as the planner LLM would write them
# ---------------------------------------------------------------------------

# A clean rates analysis (no approvals needed): the 10y-2y spread and its daily
# changes. Useful for E03/E04, determinism and replay tests.
SPREAD_PLAN: dict[str, Any] = {
    "dataset_ids": ["DGS10", "DGS2"],
    "steps": [
        {"id": "s1", "capability": "load_series", "params": {"dataset_id": "DGS10"}},
        {"id": "s2", "capability": "load_series", "params": {"dataset_id": "DGS2"}},
        {"id": "s3", "capability": "spread", "inputs": {"a": "s1", "b": "s2"}},
        {
            "id": "s4",
            "capability": "difference",
            "inputs": {"series": "s3"},
            "params": {"periods": 1},
        },
    ],
    "cannot_fulfill": False,
}

# Test-only variant of scenario 3: what a MISBEHAVING planner might emit instead
# of refusing. The validator must block it on its own (V02).
ARIMA_PLAN: dict[str, Any] = {
    "dataset_ids": ["NASDAQCOM"],
    "steps": [
        {"id": "s1", "capability": "load_series", "params": {"dataset_id": "NASDAQCOM"}},
        {
            "id": "s2",
            "capability": "arima",
            "inputs": {"series": "s1"},
            "params": {"order": [1, 0, 1], "steps_ahead": 1},
        },
    ],  # fmt: skip
    "cannot_fulfill": False,
}

# A NASDAQ volatility-regime plan, kept as a test fixture (it was an example until the
# buy-the-dip example replaced it): weekly log returns, their 12-week annualised
# rolling volatility, flags above its expanding 90th percentile, then the following
# week's simple return, compared and tested. The only plan using log returns and
# rolling_volatility, and the anchor of most guardrail tests (each mutates a copy).
VOLATILITY_PLAN: dict[str, Any] = {
    "dataset_ids": ["NASDAQCOM"],
    "steps": [
        {"id": "s1", "capability": "load_series", "params": {"dataset_id": "NASDAQCOM"}},
        {
            "id": "s2",
            "capability": "resample",
            "inputs": {"series": "s1"},
            "params": {"freq": "W", "how": "last"},
        },
        {
            "id": "s3",
            "capability": "compute_returns",
            "inputs": {"series": "s2"},
            "params": {"method": "log", "periods": 1},
        },
        {
            "id": "s4",
            "capability": "rolling_volatility",
            "inputs": {"series": "s3"},
            "params": {"window": 12, "annualize": True},
        },
        {
            "id": "s5",
            "capability": "flag_regime",
            "inputs": {"series": "s4"},
            "params": {
                "method": "expanding",
                "quantile": 0.90,
                "min_periods": 52,
                "direction": "above",
            },
        },
        {
            "id": "s6",
            "capability": "forward_return",
            "inputs": {"series": "s2"},
            "params": {"method": "simple"},
        },
        {"id": "s7", "capability": "compare_groups", "inputs": {"values": "s6", "flags": "s5"}},
        {
            "id": "s8",
            "capability": "hac_mean_difference",
            "inputs": {"outcome": "s6", "flags": "s5"},
        },
    ],  # fmt: skip
    "cannot_fulfill": False,
}

# The approvals the volatility plan needs: the return-method assumption of
# compute_returns (s3, log) and of forward_return (s6, simple).
RETURN_METHOD_APPROVAL = "V10:s3:return_method"
FORWARD_METHOD_APPROVAL = "V10:s6:return_method"
VOLATILITY_APPROVALS = [RETURN_METHOD_APPROVAL, FORWARD_METHOD_APPROVAL]

# The approval the flagship (scenario 1) needs: the return-method assumption of its
# outcome, forward_return (s6, next week's simple USD/CHF return).
SCENARIO_1_APPROVALS = ["V10:s6:return_method"]

# The approvals scenario 5 (buy the dip) needs: the return method of the weekly
# returns it flags (s3) and of the following week's return (s5), both simple.
SCENARIO_5_APPROVALS = ["V10:s3:return_method", "V10:s5:return_method"]


def raw_plan(scenario_id: str) -> dict[str, Any]:
    """A deep copy of a demo scenario's canned plan, as a dict a test may mutate."""
    return copy.deepcopy(SCENARIOS_BY_ID[scenario_id].plan)


# A full-sample threshold, kept as a test fixture for the look-ahead guardrail (V09):
# the days when the VIX was in the top 5% of its entire history, a threshold that uses
# data nobody had on the flagged day. No significance test follows it.
LOOK_AHEAD_PLAN: dict[str, Any] = {
    "dataset_ids": ["VIXCLS"],
    "steps": [
        {"id": "s1", "capability": "load_series", "params": {"dataset_id": "VIXCLS"}},
        {
            "id": "s2",
            "capability": "flag_regime",
            "inputs": {"series": "s1"},
            "params": {"method": "full_sample", "quantile": 0.95, "direction": "above"},
        },
    ],
    "cannot_fulfill": False,
}


def look_ahead_plan() -> dict[str, Any]:
    """A deep copy of LOOK_AHEAD_PLAN, as a dict a test may mutate."""
    return copy.deepcopy(LOOK_AHEAD_PLAN)


def volatility_plan() -> dict[str, Any]:
    """A deep copy of VOLATILITY_PLAN, as a dict a test may mutate."""
    return copy.deepcopy(VOLATILITY_PLAN)


def as_plan(raw: dict[str, Any], request: str = "test request") -> Plan:
    """Turn a plan dict into a Plan with defaults filled in, as the engine does."""
    return fill_defaults(Plan(plan_id="p-test", request=request, **copy.deepcopy(raw)))
