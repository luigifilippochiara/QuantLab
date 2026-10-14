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

## Plans and validation

**The LLM tool schema is the Plan schema minus `plan_id` and `request`.** The
system sets those two fields itself; asking the LLM to echo the request would
waste tokens and invite tampering. The Step definition is inlined (no `$ref`)
so the tool schema is one self-contained object.

**Strict parameter types, no coercion.** `"12"` is not an int, `true` is not an
int (Python treats bool as int, so this is checked explicitly) and NaN fails
every range check. A live LLM very rarely writes `12.0` for an integer, and if
it does, G03 says so clearly. Alternative: let Pydantic coerce, rejected because
silent conversions are exactly what an audit trail should not contain.

**Extra guardrail cases folded into existing codes.** An empty (non-refusal)
plan is G01; duplicate step ids and unknown input names are G04; the request
length and the number of distinct datasets are G06 budgets (default 2,000
characters and 5 datasets) alongside the 15-step limit.

**G08 replaces G05 for returns of a rate.** `compute_returns` on a `rate_level`
input would also be a kind mismatch (G05); only the more specific G08 is
reported because its suggestion ("use difference") is the actionable one.

**The dry run's parameter-driven metadata is declared in the registry**
(`meta_from_params`, e.g. resample sets `frequency` from `freq`). The validator
predicts output metadata from it; a test runs scenario A on synthetic data and
checks the prediction equals what the capability functions actually produced.

**Issue ids are `code:step[:key]`** (e.g. `G12:s3:return_method`). Approvals are
recorded against these ids. Any parameter edit re-validates the plan and clears
earlier approvals, so an approval always refers to the plan as it will run.

**A `scenarios.py` module (not in the spec's layout)** holds the six demo
scenarios and their canned plans as plain data. The MockPlanner, CLI demo, app
sidebar and tests all need them, so they live in one place instead of inside
planner.py. Scenario C's canned plan is deliberately bad (the validator catches
G02 and G08); scenario D's is a planner refusal (G13), so the demo shows both
lines of defence.

## Execution, lineage and monitoring

**The executor is the final gate.** `execute_plan` re-runs `validate_plan` and
refuses to start if any BLOCK exists or any APPROVAL issue id is not in the
approved set. The engine checks the same things first (to drive the UI), but
the executor does not trust its caller, so notebooks, tests and replay cannot
skip the checks.

**Output hashes use a canonical text with floats rounded to 12 significant
digits.** Series become `date,value` CSV lines, tables and test results become
JSON with sorted keys. The 12-digit rounding absorbs last-bit differences between
BLAS builds (OLS, ADF), so a replay on another machine still matches, while any
change that matters does not. The saved output file IS this canonical text, so
`shasum -a 256 runs/<id>/outputs/<file>` equals the hash in the manifest.

**R03 is checked from what `load_series` actually dropped**
(`meta.n_dropped_missing`), not from catalog counts. **R04** compares the size of
the inner join with the size of EACH input and warns if either lost more than
10%. Scenario F triggers it with the real data: DGS2 starts in 1976 and DGS10 in
1962, so the spread silently loses 22% of the 10-year history. That warning is
correct, so F completes "with warnings" rather than cleanly.

**Two extra event types**, `plan_validated` and `execution_finished`, carry the
validation and execution stage latencies needed for the per-stage latency
table; the spec's list had no event that closes those stages.

**A run's final status is its LAST terminal event.** A plan that is blocked,
then edited and executed, emits `run_failed` (blocked) and later
`run_completed` under the same run id; the dashboard counts it once, as completed.

**The manifest records the git commit and whether the tree was dirty**, in
addition to the library versions, because `registry_version` only changes when a
capability's interface changes, not when its code does.

## Planner and narrator (LLM calls)

**No forced tool call: `tool_choice: auto` plus an explicit instruction.** The
spec asked for a forced tool call. Current models (`claude-sonnet-5-5`, the
default) reject forced `tool_choice` (`any`/`tool`) with a 400. The plan schema
is still the tool's `input_schema`, so output arrives as structured JSON, and
the prompt says "respond only by calling submit_plan". If the model answers in
text, that is a G01. Strict tool mode or `output_config.format` would guarantee
schema-valid JSON, but both require closed objects (`additionalProperties:
false`) everywhere, and plan `params`/`inputs` are free-form maps. Schema
conformance is therefore enforced by our own Pydantic parse (G01), which has to
run anyway.

**No `temperature=0`.** Current models reject non-default sampling parameters
(the SDK no longer even exposes `temperature` on this endpoint). Determinism
comes from elsewhere: the plan cache (`cached` mode returns the identical plan
for an identical request, prompt, model and registry), and replay, which never
calls an LLM. Reasoning depth is set with `effort` instead (`medium` for the
planner, `low` for the narrator; configurable in `.env`).

**Refusal fallback enabled.** Both LLM calls use the API's server-side refusal
fallback (`fallbacks="default"`, beta `server-side-fallback-2026-07-01`): if the
model declines on safety grounds, the API retries on a fallback model. The
model that actually answered is recorded (`served_model`) so the run record
stays accurate. It can be removed by deleting two arguments in planner.py and
narrator.py.

**Timeouts and retries tuned for a live demo:** 90 s per call and one SDK retry,
so a stuck call fails within about 3 minutes and the presenter can switch to
cached or mock mode.

**The plan cache stores only successful planner output, and re-parses it on
load.** A broken plan is never cached (it would be served forever in cached
mode), and a cached plan goes through the same G01 parse as a fresh one. Each
cache hit gets a fresh `plan_id`. Narratives are not cached; in cached mode the
narrator calls the LLM if it can and otherwise uses the template.

**The mock planner refuses requests it does not recognise** (G13, "switch to
live mode") instead of guessing. Keyword matching is ordered most-specific
first, so the injection scenario D wins over everything else.

**Grounding check rules.** A number written with d decimals is grounded if it
is within half a unit of its last digit of an allowed value, i.e. it is a
correct rounding (so 13.86%, 14% and 0.14 all match a volatility of 0.138642,
but 0.1390 does not). Allowed values are the facts (and their x100 percentage
forms), the facts' display strings (e.g. "<0.0001"), numeric plan parameters,
numbers in the request, in warning/approval messages and in dataset titles
("10-Year"), the conventional levels 1/5/10% and 90/95/99%, and the
annualisation factors. Dates and years are allowed only within the facts' date
range. Absolute values are compared, so "-2.80" and "2.80" both match: sign
errors are a semantic problem this numeric check does not claim to catch. Known
limitation: the check proves every number is a correct rounding of SOME allowed
value, not that it is attached to the right concept.

**Facts are displayed with at least 4 significant figures and never in
scientific notation** (27204.38 shows as "27,204"; p-values with 4 decimals or
"<0.0001"), so the narrator can quote them verbatim.

## Engine, CLI and evaluation

**The engine is a `Session` dataclass with one method per stage**
(`create_plan`, `validate`, `apply_param_edits`, `approve`, `execute`,
`narrate`, `save_record`), plus `run_request` that chains them for headless
use. The app keeps the Session in `st.session_state`; the CLI and notebooks use
`run_request`. Alternative: a stateless function pipeline, rejected because
the UI genuinely needs to pause between stages for human review.

**Blocked runs are recorded immediately.** A plan that is blocked (by the
validator, the planner or missing approvals) is saved as a `blocked` run with
its issues, so refusals and blocks appear in the run history and monitoring. If
the user then fixes the plan and executes it, the same run id is overwritten as
`completed`; the dashboard counts the run once, by its last terminal event.

**Over-budget requests never reach the LLM.** The request-length budget (G06)
is checked before planning, so a 50 KB paste costs nothing.

**The CLI has an extra `trace <run_id> <step_id>` command**, and `demo` accepts
`--mode` (default `mock`, as specified). Exit codes: 0 when the outcome is the
expected one (or the run completed / all hashes match), 1 otherwise, so the
commands can be used in CI.

**Evaluation logic lives in `src/quantlab/evaluation.py`**, not in
`eval/run_eval.py`: the package's CLI cannot import a script outside the
package without dynamic imports (which the safety rules forbid). `eval/run_eval.py`
is a three-line wrapper around the same code.

**The golden set references demo scenarios by id** (`scenario: A`) instead of
repeating their request text, so it cannot drift from `scenarios.py`. In mock
mode only the six canned scenarios can match, so the mock report is a harness
check (the report says so); `--mode live` measures the real planner. A
parameter counts as correct only if EVERY step using that capability has the
expected value (both resample steps of scenario F must be weekly). Exact
sequence match is order-sensitive by definition, so equivalent plans that
load datasets in a different order lose that metric but keep a Jaccard of 1.

## Streamlit app

**`st.navigation` with explicit page titles.** `app/streamlit_app.py` is a small
router; the four pages are plain scripts in `app/pages/` and shared rendering
helpers live in `app/ui.py`. This gives readable page names ("Run history"
rather than file names) and keeps each page a top-to-bottom script.

**One colour pair for both themes.** Charts use blue `#3987e5` for series and
orange `#d95926` for flagged regimes. The pair was run through a colour-vision
validator against both Streamlit backgrounds (white and `#0e1117`): lightness,
chroma, colour-blind separation (CVD Delta E 26.8) and contrast all pass, so no
runtime theme detection is needed. Streamlit's theme detection can be wrong on
a session's first run, which made the per-theme alternative unreliable.

**One run = one execution.** After a run completes, its plan is locked in the
UI; changing anything means generating a new plan (a new run id), so a run
record always describes exactly one execution. A run stopped by a runtime
BLOCK (e.g. R02) can still be edited and re-executed under the same id.

**Approvals are ticked in the UI and recorded at execution time.** Editing any
parameter clears the engine's approvals and unticks the boxes, so an approval
always refers to the issue text of the plan that actually runs.

**The app is regression-tested headlessly** with `streamlit.testing.AppTest`
(scenario A planned, approved, executed, narrated, traced and replayed; a
blocked scenario cannot execute; all pages render).
