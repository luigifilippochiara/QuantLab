"""Tests for the data layer: FRED CSV parsing, catalog loading, hash verification."""

from pathlib import Path

import pandas as pd
import pytest

from quantlab.config import PROJECT_ROOT
from quantlab.data import (
    Catalog,
    DatasetHashMismatchError,
    describe_datasets,
    load_catalog,
    load_dataset,
    read_fred_csv,
    verify_dataset_hash,
)


def test_read_fred_csv_new_header_and_empty_missing(tmp_path: Path) -> None:
    path = tmp_path / "X.csv"
    path.write_text("observation_date,X\n2024-01-02,1.5\n2024-01-03,\n2024-01-04,2.0\n")
    series, n_rows, n_missing = read_fred_csv(path)
    assert n_rows == 3
    assert n_missing == 1
    assert list(series.values) == [1.5, 2.0]
    assert isinstance(series.index, pd.DatetimeIndex)
    assert series.dtype == float


def test_read_fred_csv_old_header_and_dot_missing(tmp_path: Path) -> None:
    path = tmp_path / "X.csv"
    # Deliberately unsorted: the loader must sort by date.
    path.write_text("DATE,X\n2024-01-04,2.0\n2024-01-02,1.5\n2024-01-03,.\n")
    series, n_rows, n_missing = read_fred_csv(path)
    assert (n_rows, n_missing) == (3, 1)
    assert list(series.index.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-04"]


def test_read_fred_csv_rejects_junk_values(tmp_path: Path) -> None:
    path = tmp_path / "X.csv"
    path.write_text("DATE,X\n2024-01-02,1.5\n2024-01-03,oops\n")
    with pytest.raises(ValueError, match="non-numeric"):
        read_fred_csv(path)


def test_read_fred_csv_rejects_duplicate_dates(tmp_path: Path) -> None:
    path = tmp_path / "X.csv"
    path.write_text("DATE,X\n2024-01-02,1.5\n2024-01-02,1.6\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_fred_csv(path)


def test_load_dataset_verifies_hash_and_drops_missing(catalog: Catalog) -> None:
    loaded = load_dataset(catalog, "NASDAQCOM")
    entry = catalog.entries["NASDAQCOM"]
    assert loaded.n_rows_raw == entry.n_rows
    assert loaded.n_missing == entry.n_missing > 0
    assert len(loaded.series) == entry.n_rows - entry.n_missing
    assert not loaded.series.isna().any()


def test_tampered_file_is_a_hard_error(catalog: Catalog) -> None:
    path = catalog.path_for("DGS10")
    path.write_text(path.read_text().replace(",5.", ",6.", 1))
    matches, _ = verify_dataset_hash(catalog, "DGS10")
    assert not matches
    with pytest.raises(DatasetHashMismatchError):
        load_dataset(catalog, "DGS10")


def test_describe_datasets_has_metadata_but_no_values(catalog: Catalog) -> None:
    described = describe_datasets(catalog, ["DGS10", "UNKNOWN"])
    assert len(described) == 1
    assert set(described[0]) == {
        "dataset_id",
        "title",
        "description",
        "units",
        "frequency",
        "kind",
        "start_date",
        "end_date",
    }


def test_committed_catalog_matches_committed_snapshot() -> None:
    """The real snapshot shipped in data/ must load and verify (no network needed)."""
    from quantlab.config import PROJECT_ROOT
    from quantlab.data import load_catalog

    real = load_catalog(PROJECT_ROOT / "data" / "catalog.json")
    assert set(real.ids()) == {"SP500", "NASDAQCOM", "VIXCLS", "DGS10", "DGS2", "DEXSZUS"}
    for dataset_id in real.ids():
        assert verify_dataset_hash(real, dataset_id)[0], dataset_id


def test_committed_catalog_display_names() -> None:
    """Display names follow the official FRED titles (long ones shortened); ids are unchanged."""
    catalog = load_catalog(PROJECT_ROOT / "data" / "catalog.json")
    titles = {d: catalog.entries[d].title for d in catalog.ids()}
    assert titles == {
        "SP500": "S&P 500 Index",
        "NASDAQCOM": "NASDAQ Composite Index",
        "VIXCLS": "CBOE Volatility Index: VIX",
        "DGS10": "10-Year U.S. Treasury Constant Maturity Yield",
        "DGS2": "2-Year U.S. Treasury Constant Maturity Yield",
        "DEXSZUS": "Swiss Francs to U.S. Dollar Spot Exchange Rate",
    }
    assert catalog.entries["SP500"].full_name == "Standard & Poor's 500"
    assert catalog.entries["DGS10"].full_name == (
        "Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity"
    )  # as written by scripts/fetch_data.py
    # The UI labels datasets by FRED id, and plans refer to dataset ids: they agree.
    assert all(e.dataset_id == e.fred_id for e in catalog.entries.values())
