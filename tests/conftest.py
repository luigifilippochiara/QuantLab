"""Shared test fixtures: synthetic datasets and isolated settings.

Tests never touch the network or the committed FRED snapshot. Instead they
build a small synthetic data folder (FRED-format CSVs + catalog.json with real
hashes) under pytest's tmp_path, using the same dataset ids as the real catalog
so the canned demo plans run unchanged.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlab.config import Settings
from quantlab.data import Catalog, file_sha256, load_catalog, read_fred_csv

# 30 years of business days: enough weekly history for scenario A's tests to
# have >= 20 observations per group even after keeping every 4th week.
N_DAYS = 7800
START = "1990-01-01"


def _business_days(n: int) -> pd.DatetimeIndex:
    """Return n consecutive business days starting at START."""
    return pd.bdate_range(START, periods=n, name="date")


def synthetic_price(n: int, seed: int, start_level: float = 100.0) -> np.ndarray:
    """Simulate a price with volatility clusters (calm 15% vs stressed 45% a year).

    The regime switches with a small daily probability, so high-volatility
    episodes come in contiguous runs, like real markets. That gives the
    flag_regime / compare_groups pipeline something to find.
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
    ("NASDAQCOM", "price_level", "Synthetic equity index", lambda: synthetic_price(N_DAYS, 1)),
    ("VIXCLS", "index_level", "Synthetic VIX", lambda: 12 + synthetic_price(N_DAYS, 2, 8) / 10),
    ("DGS10", "rate_level", "Synthetic 10-Year yield", lambda: synthetic_rate(N_DAYS, 3, 5.0)),
    ("DGS2", "rate_level", "Synthetic 2-Year yield", lambda: synthetic_rate(N_DAYS, 4, 3.5)),
    ("DEXSZUS", "price_level", "Synthetic CHF per USD", lambda: synthetic_price(N_DAYS, 5, 1.5)),
]


def build_synthetic_data_dir(data_dir: Path, missing_every: int = 50) -> Path:
    """Create snapshots/ and catalog.json for the synthetic datasets.

    Args:
        data_dir: folder to create the data in.
        missing_every: blank every k-th value (50 -> 2% missing, below the R03 threshold).

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
            }
        )
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
