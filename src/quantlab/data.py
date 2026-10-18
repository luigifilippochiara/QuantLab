"""Data layer: the dataset catalog, snapshot loading and hash verification.

Role in the architecture: all market data comes from local CSV snapshots of FRED
series listed in `data/catalog.json`. The catalog records a SHA-256 hash of each
snapshot file; every load re-computes the hash and refuses to continue if it does
not match. That makes "which data did this run use?" an exact, checkable fact.

The LLM never sees anything from this module except `describe_datasets`, which
returns metadata only (id, title, units, frequency, kind, date range), no values.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

from quantlab.types import Frequency, SeriesKind


class DatasetHashMismatchError(Exception):
    """Raised when a snapshot file's hash differs from the one recorded in the catalog."""


class DatasetEntry(BaseModel):
    """One dataset in the catalog (one FRED series snapshot).

    Fields:
        dataset_id: id used in plans (equal to the FRED id in v1).
        fred_id: the FRED series id.
        title: display name: the official FRED title, shortened when it is long.
        full_name: the full name (e.g. the complete official title), shown on hover.
        description: one-line plain description shown to users and to the LLM.
        units: units of the values (e.g. "Percent").
        frequency: observation frequency of the raw snapshot.
        kind: what the values mean (price level, rate level...).
        source_url: where the CSV was downloaded from.
        file: snapshot path, relative to the catalog file's folder.
        retrieved_at: UTC timestamp of the download.
        sha256: hash of the snapshot file's bytes.
        n_rows: rows in the file, including missing values.
        n_missing: rows whose value is missing ("" or ".").
        start_date: first date in the file.
        end_date: last date in the file.
        quote_convention: for an exchange rate, how it is quoted and so which way
            a rise or a fall goes, e.g. "USD/CHF, in Swiss francs per US dollar: ...";
            the narrator states it ("DEXSZUS is quoted as ..."). None otherwise.
    """

    # extra="forbid" so a typo in catalog.json is an error rather than silently ignored.
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
    fred_id: str
    title: str
    full_name: str
    description: str
    units: str
    frequency: Frequency
    kind: SeriesKind
    source_url: str
    file: str
    retrieved_at: str
    sha256: str
    n_rows: int
    n_missing: int
    start_date: str
    end_date: str
    quote_convention: str | None = None


@dataclass
class Catalog:
    """All datasets available to plans.

    Fields:
        entries: dataset_id -> DatasetEntry.
        base_dir: folder that the entries' `file` paths are relative to.
    """

    entries: dict[str, DatasetEntry]
    base_dir: Path

    def ids(self) -> list[str]:
        """Return all dataset ids, sorted."""
        return sorted(self.entries)

    def get(self, dataset_id: str) -> DatasetEntry | None:
        """Return the entry for `dataset_id`, or None if it is not in the catalog."""
        return self.entries.get(dataset_id)

    def path_for(self, dataset_id: str) -> Path:
        """Return the absolute path of a dataset's snapshot file."""
        return self.base_dir / self.entries[dataset_id].file


@dataclass
class LoadedDataset:
    """A loaded, hash-verified dataset.

    Fields:
        entry: its catalog entry.
        series: float values with a DatetimeIndex, missing values already dropped.
        n_rows_raw: rows in the file before dropping missing values.
        n_missing: how many rows were dropped because the value was missing.
    """

    entry: DatasetEntry
    series: pd.Series
    n_rows_raw: int
    n_missing: int


def load_catalog(path: Path) -> Catalog:
    """Read and validate the catalog file.

    Args:
        path: path of catalog.json.

    Returns:
        A Catalog whose file paths resolve relative to the catalog's folder.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = [DatasetEntry.model_validate(item) for item in raw["datasets"]]
    return Catalog(entries={e.dataset_id: e for e in entries}, base_dir=path.parent)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes.

    Args:
        path: the file to hash.

    Returns:
        64-character lowercase hex string.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_fred_csv(path: Path) -> tuple[pd.Series, int, int]:
    """Parse a FRED CSV download into a float series.

    FRED has used two header styles over time ("DATE,<ID>" and
    "observation_date,<ID>") and marks missing values either with "." or with an
    empty field; both are handled.

    Args:
        path: the CSV file.

    Returns:
        (series, n_rows_raw, n_missing): the values with missing rows dropped
        (float, DatetimeIndex named "date", sorted ascending), the number of rows
        in the file and the number of missing rows dropped.

    Raises:
        ValueError: if the file has an unexpected layout, non-numeric values or duplicate dates.
    """
    # na_values=["."] adds FRED's old missing marker; empty fields are NaN by default.
    frame = pd.read_csv(path, na_values=["."])
    if len(frame.columns) != 2:
        raise ValueError(f"{path.name}: expected 2 columns, got {list(frame.columns)}.")

    date_col = "observation_date" if "observation_date" in frame.columns else "DATE"
    if date_col not in frame.columns:
        raise ValueError(f"{path.name}: no DATE / observation_date column.")
    value_col = next(col for col in frame.columns if col != date_col)

    # A non-float column means some value was neither a number nor a missing marker.
    if not pd.api.types.is_float_dtype(frame[value_col]):
        raise ValueError(f"{path.name}: non-numeric values in column {value_col}.")

    series = pd.Series(
        frame[value_col].to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(frame[date_col], format="%Y-%m-%d"), name="date"),
        name=value_col,
    ).sort_index()
    if series.index.has_duplicates:
        raise ValueError(f"{path.name}: duplicate dates.")

    n_rows_raw = len(series)
    n_missing = int(series.isna().sum())
    return series.dropna(), n_rows_raw, n_missing


def verify_dataset_hash(catalog: Catalog, dataset_id: str) -> tuple[bool, str]:
    """Compare a snapshot file's current hash with the hash recorded in the catalog.

    Args:
        catalog: the dataset catalog.
        dataset_id: which dataset to check (must be in the catalog).

    Returns:
        (matches, actual_sha256).
    """
    actual = file_sha256(catalog.path_for(dataset_id))
    return actual == catalog.entries[dataset_id].sha256, actual


def load_dataset(catalog: Catalog, dataset_id: str) -> LoadedDataset:
    """Load one dataset after verifying its hash.

    Args:
        catalog: the dataset catalog.
        dataset_id: which dataset to load.

    Returns:
        A LoadedDataset (values, raw row count, number of missing rows dropped).

    Raises:
        KeyError: if the dataset is not in the catalog.
        DatasetHashMismatchError: if the file changed since the catalog was written.
    """
    entry = catalog.entries[dataset_id]
    matches, actual = verify_dataset_hash(catalog, dataset_id)
    if not matches:
        # A hard error on purpose: silently using different data would make every
        # result and every replay untrustworthy.
        raise DatasetHashMismatchError(
            f"{dataset_id}: file hash {actual[:12]}... does not match catalog "
            f"hash {entry.sha256[:12]}...; the snapshot was modified."
        )
    series, n_rows_raw, n_missing = read_fred_csv(catalog.path_for(dataset_id))
    return LoadedDataset(
        entry=entry, series=series.rename(dataset_id), n_rows_raw=n_rows_raw, n_missing=n_missing
    )


def describe_datasets(catalog: Catalog, dataset_ids: list[str] | None = None) -> list[dict]:
    """Return LLM-safe metadata for datasets: no data values, only descriptions.

    Args:
        catalog: the dataset catalog.
        dataset_ids: which datasets to describe; None or empty means all.

    Returns:
        One dict per dataset with id, title, description, units, frequency, kind,
        start_date and end_date. Unknown ids are skipped (the validator reports them).
    """
    ids = dataset_ids or catalog.ids()
    described = []
    for dataset_id in ids:
        entry = catalog.get(dataset_id)
        if entry is None:
            continue
        described.append(
            {
                "dataset_id": entry.dataset_id,
                "title": entry.title,
                "description": entry.description,
                "units": entry.units,
                "frequency": entry.frequency.value,
                "kind": entry.kind.value,
                "start_date": entry.start_date,
                "end_date": entry.end_date,
            }
        )
    return described
