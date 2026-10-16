"""Download the FRED snapshot and write data/catalog.json.

Run once (needs network access):

    uv run python scripts/fetch_data.py          # every series
    uv run python scripts/fetch_data.py SP500    # only the listed series


Each series is saved byte-for-byte as served by FRED's keyless CSV endpoint to
data/snapshots/<ID>.csv, and the catalog records its SHA-256 hash, row counts and
date range. The snapshot is committed to git so every run is reproducible
without network access; re-running this script creates a NEW data version
(new hashes), which old run records will then report as a mismatch on replay.
Fetching only some series leaves the other catalog entries (and files) untouched.
"""

import json
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from quantlab.config import load_settings
from quantlab.data import file_sha256, read_fred_csv

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

# The series QuantLab ships with. `kind` drives the validator's guardrails, e.g.
# rate_level series must be differenced, never turned into returns. `title` is the
# display name (the official FRED title, shortened when it is long) and `full_name`
# the full name shown on hover.
SERIES_TO_FETCH: list[dict[str, str]] = [
    {
        "fred_id": "SP500",
        "title": "S&P 500 Index",
        "full_name": "Standard & Poor's 500",
        "description": "S&P 500 Index, daily close (index level); FRED keeps 10 years.",
        "units": "Index",
        "kind": "price_level",
    },
    {
        "fred_id": "NASDAQCOM",
        "title": "NASDAQ Composite Index",
        "full_name": "NASDAQ Composite Index",
        "description": "NASDAQ Composite equity index, daily close (index level).",
        "units": "Index, Feb 5 1971 = 100",
        "kind": "price_level",
    },
    {
        "fred_id": "VIXCLS",
        "title": "CBOE Volatility Index: VIX",
        "full_name": "CBOE Volatility Index: VIX",
        "description": "VIX implied volatility index, daily close; a level, not a tradable price.",
        "units": "Index (annualised implied volatility, percent)",
        "kind": "index_level",
    },
    {
        "fred_id": "DGS10",
        "title": "10-Year U.S. Treasury Constant Maturity Yield",
        "full_name": (
            "Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity, "
            "Quoted on an Investment Basis"
        ),
        "description": "10-year US Treasury constant-maturity yield, daily, in percent.",
        "units": "Percent",
        "kind": "rate_level",
    },
    {
        "fred_id": "DGS2",
        "title": "2-Year U.S. Treasury Constant Maturity Yield",
        "full_name": (
            "Market Yield on U.S. Treasury Securities at 2-Year Constant Maturity, "
            "Quoted on an Investment Basis"
        ),
        "description": "2-year US Treasury constant-maturity yield, daily, in percent.",
        "units": "Percent",
        "kind": "rate_level",
    },
    {
        "fred_id": "DEXSZUS",
        "title": "Swiss Francs to U.S. Dollar Spot Exchange Rate",
        "full_name": "Swiss Francs to U.S. Dollar Spot Exchange Rate",
        "description": "CHF per USD spot exchange rate, daily noon buying rate in New York.",
        "units": "Swiss Francs to One U.S. Dollar",
        "kind": "price_level",
    },
]


def download(url: str, destination: Path) -> None:
    """Download `url` and write the raw bytes to `destination`.

    Args:
        url: the URL to fetch.
        destination: file to write.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "QuantLab/1.0 (snapshot fetch)"})
    with urllib.request.urlopen(request, timeout=60) as response:
        destination.write_bytes(response.read())


def fetch_series(item: dict[str, str], snapshots_dir: Path) -> dict[str, object]:
    """Download one series and describe it as a catalog entry.

    Args:
        item: one element of SERIES_TO_FETCH.
        snapshots_dir: folder for the CSV files.

    Returns:
        The catalog entry (hash, counts and date range of the downloaded file).
    """
    series_id = item["fred_id"]
    url = FRED_CSV_URL.format(series_id=series_id)
    path = snapshots_dir / f"{series_id}.csv"
    print(f"Downloading {series_id} ...", flush=True)
    download(url, path)

    # Parse with the same reader the engine uses, so the catalog's counts and
    # date range describe exactly what a load will see.
    series, n_rows_raw, n_missing = read_fred_csv(path)
    entry: dict[str, object] = {
        "dataset_id": series_id,
        "fred_id": series_id,
        "title": item["title"],
        "full_name": item["full_name"],
        "description": item["description"],
        "units": item["units"],
        "frequency": "D",
        "kind": item["kind"],
        "source_url": url,
        "file": f"snapshots/{series_id}.csv",
        "retrieved_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256": file_sha256(path),
        "n_rows": n_rows_raw,
        "n_missing": n_missing,
        "start_date": series.index[0].strftime("%Y-%m-%d"),
        "end_date": series.index[-1].strftime("%Y-%m-%d"),
    }
    print(f"  {n_rows_raw} rows, {n_missing} missing, {entry['start_date']} to {entry['end_date']}")
    return entry


def main(only: list[str]) -> int:
    """Fetch the series (all, or only those named), write the snapshot files and the catalog.

    Args:
        only: FRED ids to fetch; empty means every series in SERIES_TO_FETCH.

    Returns:
        Process exit code (0 on success, 1 for an unknown id).
    """
    known = [item["fred_id"] for item in SERIES_TO_FETCH]
    unknown = sorted(set(only) - set(known))
    if unknown:
        print(f"Unknown series {unknown}; known: {known}")
        return 1

    settings = load_settings()
    snapshots_dir = settings.data_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    # Start from the current catalog so a partial fetch keeps the other entries.
    existing: dict[str, dict[str, object]] = {}
    if settings.catalog_path.exists():
        current = json.loads(settings.catalog_path.read_text(encoding="utf-8"))
        existing = {entry["dataset_id"]: entry for entry in current["datasets"]}

    entries = []
    for item in SERIES_TO_FETCH:
        if not only or item["fred_id"] in only:
            entries.append(fetch_series(item, snapshots_dir))
        elif item["fred_id"] in existing:
            entries.append(existing[item["fred_id"]])

    catalog = {
        "description": "QuantLab dataset catalog: FRED snapshots with file hashes.",
        "datasets": entries,
    }
    settings.catalog_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {settings.catalog_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
