# Data

QuantLab analyses a small, fixed snapshot of public daily series from
[FRED](https://fred.stlouisfed.org) (Federal Reserve Bank of St. Louis).

| dataset_id | FRED series | What it is | Kind | History in snapshot |
|---|---|---|---|---|
| `SP500` | [SP500](https://fred.stlouisfed.org/series/SP500) | S&P 500 index, daily close | `price_level` | 2016-10-17 to 2026-10-15 |
| `NASDAQCOM` | [NASDAQCOM](https://fred.stlouisfed.org/series/NASDAQCOM) | NASDAQ Composite Index, daily close | `price_level` | 1971-02-05 to 2026-10-13 |
| `VIXCLS` | [VIXCLS](https://fred.stlouisfed.org/series/VIXCLS) | CBOE VIX, daily close | `index_level` | 1990-01-02 to 2026-10-13 |
| `DGS10` | [DGS10](https://fred.stlouisfed.org/series/DGS10) | 10-year Treasury constant-maturity yield (%) | `rate_level` | 1962-01-02 to 2026-10-09 |
| `DGS2` | [DGS2](https://fred.stlouisfed.org/series/DGS2) | 2-year Treasury constant-maturity yield (%) | `rate_level` | 1976-06-01 to 2026-10-09 |
| `DEXSZUS` | [DEXSZUS](https://fred.stlouisfed.org/series/DEXSZUS) | Swiss francs per US dollar, spot | `price_level` | 1971-01-04 to 2026-10-09 |

**Retrieved:** 2026-10-14, SP500 on 2026-10-16 (UTC timestamps per series are in
`catalog.json`). FRED distributes only the last 10 years of the S&P 500.

## Files

- `snapshots/<ID>.csv`: the CSV exactly as served by FRED's keyless endpoint
  `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ID>`. Header is
  `observation_date,<ID>`; missing values (exchange holidays) are empty fields.
  Older FRED downloads used `DATE` and `.`; the loader accepts both.
- `catalog.json`: one entry per dataset with title, units, frequency, kind,
  source URL, retrieval time, **SHA-256 of the file**, row and missing counts and
  date range.

The snapshot is committed so that a clean clone can run every analysis, test
and replay without network access. Every load re-hashes the file and refuses to
run if it no longer matches the catalog (guardrail R01).

## Refreshing

```bash
uv run python scripts/fetch_data.py
```

This overwrites the snapshot and catalog with today's data, which is a new data
version: older run records will report a dataset hash mismatch on replay, as
intended. Naming series (`uv run python scripts/fetch_data.py SP500`) fetches only
those and leaves the other files and catalog entries unchanged; that is how
SP500 was added.

## Licence notes

- `DGS10`, `DGS2` (Federal Reserve Board, H.15 release) and `DEXSZUS` (Federal
  Reserve Board, H.10 release) are marked **public domain** on FRED.
- `VIXCLS` is marked on FRED as *Copyrighted: Citation Required*
  ("Copyright, Chicago Board Options Exchange, Inc. Reprinted with permission").
- `NASDAQCOM` is marked on FRED as *Copyrighted: Pre-Approval Required*
  (Copyright NASDAQ OMX Group, Inc.).
- `SP500` is marked on FRED as *Copyrighted: Pre-Approval Required*
  (Copyright S&P Dow Jones Indices LLC).

The snapshot is included so that every analysis can be reproduced exactly. Before
publishing this repository or redistributing the files, check the
[FRED Terms of Use](https://fred.stlouisfed.org/legal/) and the series notes;
the NASDAQ and S&P 500 series in particular may need the copyright holder's
approval. No public-domain equity index with comparable history is available on FRED.
