# Decisions

Choices made where the spec (docs/SPEC.md) was silent, ambiguous, or did not
work in practice. One short paragraph each: what, why, alternatives considered.

## Project setup

**Python 3.12+, not 3.11+.** The current numpy release (2.5) requires Python
3.12, and the repo already pinned 3.12 in `.python-version`. `requires-python`
is `>=3.12`, which still satisfies "3.11+". Alternative: pin older numpy/pandas
to keep 3.11, rejected because it means older libraries for no practical gain.

**Exact pins in pyproject.toml, plus uv.lock.** Direct dependencies use `==`
so the versions recorded in every run record's `environment` section are the
versions a fresh `uv sync` installs; uv.lock pins the transitive ones.
Alternative: `>=` ranges with the lockfile as the only pin, rejected because a
`uv lock --upgrade` would then silently change numerical libraries.

**src layout, package `quantlab`.** Installed in editable mode by `uv sync`, so
the app, CLI, notebooks, eval and tests all `import quantlab` the same way.

**Work on a branch.** Phase commits go to the `quantlab-v1` branch rather than
directly to `main`, so the whole v1 build can be reviewed and merged in one step
(`git merge --ff-only quantlab-v1`).

## Data

**dataset_id equals the FRED id** (e.g. `NASDAQCOM`). Unambiguous, matches the
file names and the FRED website. The catalog still keeps a separate `fred_id`
field so non-FRED sources can be added later. Alternative: friendly ids like
`nasdaq`, rejected as one more mapping to explain.

**Snapshots are stored byte-for-byte as FRED serves them**, and the catalog
hashes those bytes. `.gitattributes` marks the CSVs `-text` so git never
rewrites line endings, otherwise a Windows checkout would change the hashes.
Alternative: normalise to our own CSV format, rejected because "this is exactly
what FRED returned" is the simplest provenance story.

**Two extra catalog fields:** `description` (a plain one-liner shown to users
and the LLM) and `file` (snapshot path relative to the catalog), on top of the
fields listed in the spec.

**All five specified series were kept.** All download with long history
(1962-1990 starts). FRED marks NASDAQCOM as "Copyrighted: Pre-Approval
Required" and VIXCLS as "Citation Required"; there is no public-domain equity
index with comparable history on FRED, so NASDAQCOM stays and the licence
terms are documented in `data/README.md` (check them before publishing the repo).

## Configuration

**Default LLM mode is `cached` when an API key is set, `mock` otherwise**, so
the app, CLI and notebooks work out of the box without a key. Overridable with
`QUANTLAB_LLM_MODE` and per request in the UI and CLI.
