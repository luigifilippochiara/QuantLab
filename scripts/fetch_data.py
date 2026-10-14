"""Download the FRED snapshot and write data/catalog.json.

Run once (needs network access):

    uv run python scripts/fetch_data.py

Each series is saved byte-for-byte as served by FRED's keyless CSV endpoint to
data/snapshots/<ID>.csv, and the catalog records its SHA-256 hash, row counts and
date range. The snapshot is committed to git so every run is reproducible
without network access; re-running this script creates a NEW data version
(new hashes), which old run records will then report as a mismatch on replay.
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
# rate_level series must be differenced, never turned into returns.
SERIES_TO_FETCH: list[dict[str, str]] = [
    {
        "fred_id": "NASDAQCOM",
        "title": "NASDAQ Composite Index",
        "description": "NASDAQ Composite equity index, daily close (index level).",
        "units": "Index, Feb 5 1971 = 100",
        "kind": "price_level",
    },
    {
        "fred_id": "VIXCLS",
        "title": "CBOE Volatility Index: VIX",
        "description": "VIX implied volatility index, daily close; a level, not a tradable price.",
        "units": "Index (annualised implied volatility, percent)",
        "kind": "index_level",
    },
    {
        "fred_id": "DGS10",
        "title": "Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity",
        "description": "10-year US Treasury constant-maturity yield, daily, in percent.",
        "units": "Percent",
        "kind": "rate_level",
    },
    {
        "fred_id": "DGS2",
        "title": "Market Yield on U.S. Treasury Securities at 2-Year Constant Maturity",
        "description": "2-year US Treasury constant-maturity yield, daily, in percent.",
        "units": "Percent",
        "kind": "rate_level",
    },
    {
        "fred_id": "DEXSZUS",
        "title": "Swiss Francs to U.S. Dollar Spot Exchange Rate",
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


def main() -> int:
    """Fetch every series, write the snapshot files and the catalog.

    Returns:
        Process exit code (0 on success).
    """
    settings = load_settings()
    snapshots_dir = settings.data_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for item in SERIES_TO_FETCH:
        series_id = item["fred_id"]
        url = FRED_CSV_URL.format(series_id=series_id)
        path = snapshots_dir / f"{series_id}.csv"
        print(f"Downloading {series_id} ...", flush=True)
        download(url, path)

        # Parse with the same reader the engine uses, so the catalog's counts and
        # date range describe exactly what a load will see.
        series, n_rows_raw, n_missing = read_fred_csv(path)
        entries.append(
            {
                "dataset_id": series_id,
                "fred_id": series_id,
                "title": item["title"],
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
        )
        print(
            f"  {n_rows_raw} rows, {n_missing} missing, "
            f"{entries[-1]['start_date']} to {entries[-1]['end_date']}"
        )

    catalog = {
        "description": "QuantLab dataset catalog: FRED snapshots with file hashes.",
        "datasets": entries,
    }
    settings.catalog_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {settings.catalog_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
