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

## Capabilities and registry

**Capability functions take natural keyword arguments**, e.g.
`compute_returns(series, method="simple", periods=1)`: inputs are Artifacts named
as in the registry, parameters are plain values. The executor calls
`spec.func(**inputs, **params)`; a test checks every function's signature equals
its spec's inputs + params. `load_series` is the one function that touches
files, so its spec sets `reads_catalog=True` and the executor passes the
catalog. Alternative: a uniform `func(inputs: dict, params: dict)` signature,
rejected because it is harder to read and to call from notebooks and tests.

**Registry specs are frozen dataclasses; plans are Pydantic models.** Specs are
written by us, hold function references and are never parsed from outside
input; plans come from an LLM and must be parsed strictly. Using Pydantic only
where untrusted data enters keeps that boundary visible.

**SERIES and FLAGS artifacts never contain NaN.** Warm-up rows (the first
window-1 rolling values, the last h forward returns, the first min_periods
expanding flags) are dropped instead of kept as NaN, so `n_obs` always counts
usable values and every join is a plain inner join. Expanding flags are not set
to False during the warm-up because that would put warm-up weeks into the
"normal" group.

**`dataset_id` is a free string checked by G07, not a registry `choices` list.**
Putting the catalog's ids into the registry would change `registry_version()`
every time the data is refreshed. The LLM still sees the valid ids, in the
dataset section of its prompt.

**Allowed input kinds where the spec was silent.** `resample` and `difference`
take levels only (resampling returns with "last" would silently produce wrong
weekly returns); `flag_regime` refuses forward returns (flagging on future
information); `spread` takes rates, returns, changes or volatilities. Anything
else is a G05 TYPE_MISMATCH with a suggestion.

**Multi-period returns.** `compute_returns(periods=k)` records `periods=k` in the
metadata. `rolling_volatility` then annualises by sqrt(factor / k) (there are
only 52/k k-week returns per year), and `forward_return` refuses k > 1 because
compounding overlapping k-period returns is wrong.

**`resample` drops an incomplete final period and refuses upsampling.** pandas
labels a period by its end date, so data ending on a Tuesday produced a bar
dated the coming Friday (a future date) holding a partial week. Monthly
resampling of weekly data is allowed; weekly from monthly raises an error.

**Flags are strictly above the threshold** (`>`), and the expanding threshold
uses pandas' default linear interpolation (the same as `numpy.quantile`).

**min_obs rules the spec did not give**: returns/differences need 2 x periods,
`flag_regime` 2 x min_periods, `forward_return` 2 x horizon, `summary_stats` 4,
`adf_test` 50. `ols_hac` uses the same 20-per-group rule as `compare_groups`,
since a dummy regression with few flagged observations has the same power problem.

**Flagged episodes are counted on the sample actually tested** (after the
non-overlapping subsample in `compare_groups`), since that is what determines
the test's power.

**TEST_RESULT values are `{"test", "statistic", "p_value", "details"}`.** For
`ols_hac` the statistic is the t-statistic of the flag coefficient.
