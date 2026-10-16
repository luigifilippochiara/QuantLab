# Design decisions

The design decisions behind QuantLab: choices made where the v1 specification
(docs/SPEC.md) was silent, ambiguous, or did not work in practice, and the
rationale for later changes to the product. One short paragraph each: what, why,
alternatives considered.

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
terms are documented in `data/README.md` (check them before redistributing the data).

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
window-1 rolling values, the first min_periods expanding flags) are dropped
instead of kept as NaN, so `n_obs` always counts usable values and every join is
a plain inner join. Expanding flags are not set to False during the warm-up
because a day without enough history is unknown, not calm.

**`dataset_id` is a free string checked by G07, not a registry `choices` list.**
Putting the catalog's ids into the registry would change `registry_version()`
every time the data is refreshed. The LLM still sees the valid ids, in the
dataset section of its prompt.

**Allowed input kinds where the spec was silent.** `resample` and `difference`
take levels only (resampling returns with "last" would silently produce wrong
weekly returns); `spread` takes rates, returns, changes or volatilities;
`flag_regime` and `summary_stats` take any series (but not flags). Anything else
is a G05 TYPE_MISMATCH with a suggestion.

**Multi-period returns.** `compute_returns(periods=k)` records `periods=k` in the
metadata. `rolling_volatility` then annualises by sqrt(factor / k) (there are
only 52/k k-week returns per year).

**`resample` drops an incomplete final period and refuses upsampling.** pandas
labels a period by its end date, so data ending on a Tuesday produced a bar
dated the coming Friday (a future date) holding a partial week. Monthly
resampling of weekly data is allowed; weekly from monthly raises an error.

**Flags are strictly above the threshold** (`>`), and the expanding threshold
uses pandas' default linear interpolation (the same as `numpy.quantile`).

**min_obs rules the spec did not give**: returns/differences need 2 x periods,
`flag_regime` 2 x min_periods, `summary_stats` 4.

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
predicts output metadata from it; a test runs several plans on synthetic data
and checks the prediction equals what the capability functions actually produced.

**Issue ids are `code:step[:key]`** (e.g. `G12:s2:return_method`). Approvals are
recorded against these ids. Any parameter edit re-validates the plan and clears
earlier approvals, so an approval always refers to the plan as it will run.

**A `mock_scenarios.py` module (not in the spec's layout)** holds the example
workflows and their canned plans as plain data. The MockPlanner, `cli demo`, the
app sidebar, the tests and the golden set all need them, so they live in one
place instead of inside planner.py. See "Example workflows" below for the current set.

## Execution, lineage and monitoring

**The executor is the final gate.** `execute_plan` re-runs `validate_plan` and
refuses to start if any BLOCK exists or any APPROVAL issue id is not in the
approved set. The engine checks the same things first (to drive the UI), but
the executor does not trust its caller, so notebooks, tests and replay cannot
skip the checks.

**Output hashes use a canonical text with floats rounded to 12 significant
digits.** Series become `date,value` CSV lines, tables become JSON with sorted
keys. The 12-digit rounding absorbs last-bit differences between CPU builds and
library versions, so a replay on another machine still matches, while any
change that matters does not. The saved output file IS this canonical text, so
`shasum -a 256 runs/<id>/outputs/<file>` equals the hash in the manifest.

**R03 is checked from what `load_series` actually dropped**
(`meta.n_dropped_missing`), not from catalog counts. **R04** compares the size of
the inner join with the size of EACH input and warns if either lost more than
10%. The 10y-2y spread triggers it with the real data: DGS2 starts in 1976 and
DGS10 in 1962, so the spread silently loses 22% of the 10-year history. That
warning is correct, so such a run completes "with warnings" rather than cleanly.

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

**Timeouts and retries tuned for interactive use:** 120 s per call and one SDK
retry, so a stuck call fails within about 4 minutes and the user can switch to
cached or mock mode. (Raised from 90 s when OpenAI reasoning models were
added, which can take longer at medium effort.)

**The plan cache stores only successful planner output, and re-parses it on
load.** A broken plan is never cached (it would be served forever in cached
mode), and a cached plan goes through the same G01 parse as a fresh one. Each
cache hit gets a fresh `plan_id`. Narratives are not cached; in cached mode the
narrator calls the LLM if it can and otherwise uses the template.

**The mock planner refuses requests it does not recognise** (G13, "switch to
live mode") instead of guessing. Keyword matching is ordered most-specific
first, so the prompt-injection example wins over everything else.

**Grounding check rules.** A number written with d decimals is grounded if it
is within half a unit of its last digit of an allowed value, i.e. it is a
correct rounding (so 13.86%, 14% and 0.14 all match a volatility of 0.138642,
but 0.1390 does not). Allowed values are the facts (and their x100 percentage
forms), the numbers in the facts' display strings, numeric plan parameters,
numbers in the request, in warning/approval messages and in dataset titles
("10-Year"), and the annualisation factors. Dates and years are allowed only within the facts' date
range. Absolute values are compared, so "-2.80" and "2.80" both match: sign
errors are a semantic problem this numeric check does not claim to catch. Known
limitation: the check proves every number is a correct rounding of SOME allowed
value, not that it is attached to the right concept.

**Facts are displayed with at least 4 significant figures and never in
scientific notation** (27204.38 shows as "27,204", 0.000563124 as "0.0005631"),
so the narrator can quote them verbatim.

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

**The golden set references example workflows by id** (`scenario: volatility_flag_success`)
instead of repeating their request text, so it cannot drift from `mock_scenarios.py`.
In mock mode only the five canned examples can match, so the mock report is a harness
check (the report says so); `--mode live` measures the real planner. A
parameter counts as correct only if EVERY step using that capability has the
expected value (if a plan resamples two series, both must be weekly). Only
parameter values the request implies are scored, not choices left to the
planner (such as a window the request does not state). Exact
sequence match is order-sensitive by definition, so equivalent plans that
load datasets in a different order lose that metric but keep a Jaccard of 1.

## Streamlit app

**`st.navigation` with explicit page titles.** `app/streamlit_app.py` is a small
router; the four pages are plain scripts in `app/pages/` and shared rendering
helpers live in `app/ui.py`. This gives readable page names ("Run history"
rather than file names) and keeps each page a top-to-bottom script.

**The sidebar title is drawn above a page menu made of `st.page_link`s.**
`st.navigation` always draws its own menu at the very top of the sidebar, so
nothing can be placed above it. The router therefore hides that menu
(`position="hidden"`), then draws `st.header("QuantLab")`, the same call the pages
use for "Datasets" and "LLM mode", so it matches their font and size by
construction, followed by one `st.page_link` per `st.Page` (same titles and icons),
and finally runs the page, whose own sidebar content comes below. It lives in
`streamlit_app.py`, so all four pages share it. Alternative: CSS that moves or
prefixes the native menu, rejected as fragile: it depends on Streamlit's internal
markup, which can change between versions.

**No dataset is preselected.** The Datasets picker starts empty, so the analyst
chooses the data instead of starting from a NASDAQ default; an example workflow
fills in its own datasets. With nothing selected, the planner still sees every
catalog dataset's metadata and is told that none was selected.

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
(example 1 planned, edited, approved, executed, narrated, traced and replayed;
blocked examples cannot execute; all pages render).

## Notebooks and documentation

**Notebooks are committed executed, in mock mode, on the real snapshot.** Each
of the eight notebooks starts with a markdown cell on what its layer does and
why, imports everything from `quantlab` (no logic is copied) and is executed with
`jupyter nbconvert --execute --inplace`, so a reader sees outputs without
running anything. Outputs avoid absolute paths. Notebook 05 switches to live
mode automatically when the configured provider's API key is set. They write their runs to
`runs/` like the app and the CLI. No charts are embedded, to keep the files small.

**The README's guardrail table is checked against the code** (`tests/test_docs.py`),
as are the capability names, so the documentation cannot silently drift.

**Evaluation reports are not committed** (`eval/reports/` is git-ignored except
for a `.gitkeep`): they are timestamped outputs; regenerate them with
`quantlab.cli eval`.

**Streamlit telemetry is off** (`.streamlit/config.toml`,
`browser.gatherUsageStats = false`): by default Streamlit sends usage statistics
to a third party, which an internal bank tool should not do.

## LLM provider: OpenAI added (default) alongside Anthropic

**One setting, `QUANTLAB_LLM_PROVIDER` (`openai` by default, or `anthropic`),
chooses the API for both LLM calls.** OpenAI is the provider used for live
operation. Rather than deleting the Anthropic implementation, both providers
sit side by side in `planner.py` (planner classes, client factories, error
messages) and `narrator.py` (one small function per provider returning the same
`LLMReply`). Both get the same system prompt, the same plan schema and the same
validation, so swapping providers changes nothing downstream. The rule "the LLM
appears only in the planner and narrator" still holds: only `planner.py` imports
the SDKs, enforced by a test. Alternative: replace Anthropic entirely, rejected
because the spec names it and the switch costs little code.

**OpenAI uses the Responses API, not Chat Completions.** OpenAI's reasoning
guide recommends Responses for reasoning models and states that Chat Completions
does not support function calling with GPT-6.1 Sol. Function
tools in Responses are flat (`{"type": "function", "name", "parameters",
"strict"}`); the plan arrives as a JSON string in a `function_call` output item,
which is parsed and then validated like any other plan. An invalid JSON string
is a G01.

**Default OpenAI model `gpt-6-luna`, for both the planner and the narrator.**
The first default was `gpt-6.1-sol`, OpenAI's "balanced intelligence and cost"
model (per its models page in October 2026). The default moved to `gpt-6-luna`,
the cost-efficient GPT-6 model, after it had run live for both calls without
problems; the narrator follows the planner's model unless
`QUANTLAB_NARRATOR_MODEL` is set. `gpt-6.1-sol` remains available as an
alternative through `QUANTLAB_PLANNER_MODEL` if planning quality on free-form
requests requires it.

**Same request design as on Anthropic: `tool_choice="auto"` plus an explicit
instruction, `strict: False`, no `temperature`.** OpenAI's docs do not confirm
that the GPT-6 reasoning models accept a forced tool choice or a custom
temperature, and a rejected parameter would break live sessions; `"required"`
or a forced `{"type": "function", "name": "submit_plan"}` is a one-line change
to try against the live API. Strict mode requires closed objects, and plan
params are free-form maps. Reasoning depth uses `reasoning={"effort": ...}` with
the existing `medium` (planner) / `low` (narrator) settings, which all three
GPT-6 models accept.

**OpenAI planner output cap 25,000 tokens.** The cap includes reasoning tokens,
and OpenAI's reasoning guide recommends reserving at least 25,000 at first. An
`incomplete` response (e.g. `max_output_tokens` reached) is a G01 for the
planner and a template fallback for the narrator. Anthropic keeps 16,000.

**No server-side refusal fallback on OpenAI.** That feature is Anthropic's; on
OpenAI a refusal simply produces no `submit_plan` call (G01), and the narrator
falls back to the template.

**`demo --mode live` checks only the outcome status.** With a real LLM a
example may be blocked for a different reason than the canned plan (e.g. a
refusal G13 for example 5 instead of G09, which the spec explicitly allows). In
mock mode the issue codes are still checked exactly. Running `cli demo` in live
mode also pre-loads the plan cache for later cached-mode sessions.

**Verification of the OpenAI path.** Besides unit tests with simulated clients,
the OpenAI path was run end to end with the real `openai` SDK against a local
stand-in server returning Responses-API JSON: the request carried the intended
parameters, the real response objects parsed, the run completed and the
narrative was grounded. The real API has since been used with `gpt-6-luna`
(planner and narrator both worked; an unknown model id gave the expected G01
with "API error 404").

## Descriptive capabilities only

**Statistical tests removed: `compare_groups` (Welch t / Mann-Whitney),
`ols_hac` (Newey-West regression) and `adf_test`; and `forward_return` with
them.** The scope was narrowed to simple, descriptive analyses, which analysts
can interpret correctly without a quant in the loop and reviewers can verify by
reading. `forward_return` only existed to feed the regime-comparison tests;
without them it was a half workflow, and a "future return" column is easy to
confuse with a prediction, which QuantLab must decline. The menu became eight
descriptive capabilities (nine once `compare_groups` returned in descriptive
form, below). Alternative: keep the tests but leave them out of the example
workflows, rejected because the planner would still see and use them and the
code would still need maintaining.

**Guardrails that only applied to tests were retired, without renumbering.**
G11 OVERLAPPING_OBSERVATIONS (t-tests on overlapping forward returns), R05
LOW_EFFECTIVE_SAMPLE (test power from few flagged episodes) and R02's per-group
minimum are gone. The other codes keep their numbers, leaving a gap at G11.
Renumbering was tried first and reverted: approvals are recorded by issue id
(e.g. `G12:s2:return_method`), so renumbering made existing run records fail to
replay even though they only used capabilities that still exist. Never
reusing a code is the usual rule for exactly this reason.

**scipy and statsmodels removed from the dependencies.** Nothing uses them any
more. The summary-statistics test now checks skew and excess kurtosis against
their textbook formulas instead of scipy, which is also a more independent check.

**An intermediate set of examples preceded the current workflows**: the average
daily S&P 500 Index return, daily high-volatility periods in the NASDAQ, a forecast
request whose canned plan named a non-existent `forecast` capability (blocked by
G02), the prompt injection, and the 10y-2y spread. It is superseded by the
example workflows below.

**SP500 added to the catalog**, as the reference equity index analysts most
often ask about. FRED distributes only its last 10 years and marks it "Copyrighted:
Pre-Approval Required", like NASDAQCOM (documented in `data/README.md`).
NASDAQCOM stays for its 55-year history, which gives the high-volatility
analysis many episodes. `scripts/fetch_data.py SP500` fetched only the new
series, so the existing snapshot files and hashes did not change.

**Narrator and prompt versions.** The narrator's rule on reading p-values became
"describe what was computed; no causes, forecasts or significance claims", and
conventional significance levels are no longer accepted numbers in the
grounding check. The prompt versions were bumped (`planner-v2`, `narrator-v2`,
`canned-scenarios-v2`); together with the new registry version this means plans
cached before the change are never reused.

## Example workflows

**Five ready-made requests and the audit trail.** The example workflows follow
one realistic task, a market-risk analyst identifying periods of unusual market
behaviour: the core analysis (1, high-volatility weeks, `volatility_flag_success`)
and four built-in protections, against look-ahead (2,
`volatility_flag_lookahead`), out-of-scope requests (3, `sarima_forecast_block`),
prompt injection (4, `prompt_injection_block`) and misaligned frequencies (5,
`frequency_mismatch_block`). The trace of example 1's `flag_regime` step shows
the audit trail. They replace the earlier examples everywhere (sidebar,
`cli demo`, golden set); the ids are kept as stable identifiers for the tests,
the golden set and the CLI. Tests that need a clean rates plan or a misbehaving
planner keep their own plans in `tests/conftest.py` (`SPREAD_PLAN`,
`SARIMA_PLAN`) rather than in `mock_scenarios.py`, so the app shows only the
example set. Examples are matched by exact text, id, number ("1".."5") or keywords.

**`compare_groups` is back, as a descriptive capability.** Example 5's plan ends
in `compare_groups(values, flags)`, which had been removed with the statistical
tests; without it the validator would report G02 (unknown capability) and never
reach the frequency check. It returns the mean of the values in flagged and in
normal periods, the group sizes and the difference, and nothing else: no p-value
or test statistic, no scipy, consistent with the earlier "no statistical tests"
decision. Its assumptions say so explicitly, and the narrator template calls the
result descriptive. Alternative: make example 5 use `spread` of two
differently-sampled series, rejected because the workflow it represents compares
returns across VIX regimes.

**Example 3 is a planner refusal; the G02 check is tested separately.** The canned
plan refuses (G13) and lists what the registry can do, as a well-behaved LLM
should. A test feeds a misbehaving planner's plan with a `sarima` step
through the engine (live mode, fake client) and asserts the validator blocks it
with G02 and nothing runs.

**Example 4's "no shell" test allows the run record's git queries.** The engine
saves even blocked runs, and the manifest records the git commit and dirty flag
through two fixed, read-only `git` commands (`subprocess.run` with an argument
list, never a shell). The test therefore asserts that `os.system` and `os.popen`
are never called and that every process started is one of those `git` commands,
with nothing from the request in its arguments. Alternative: read `.git` files
directly to avoid subprocess entirely, rejected because it loses the dirty flag
and adds parsing code.

**Plain-English G09 and G10.** G10's message now explains that a full-sample
threshold judges past weeks (or days, months: taken from the input's frequency)
using information from the future, and suggests `method='expanding'`. G09 names
which input is too frequent, the dataset it comes from, and the frequency to
resample to (e.g. "Resample VIXCLS to weekly right after load_series").

**`cli demo` auto-approves only examples 1 and 2** (`Scenario.auto_approve`), prints
the example id, status, issue codes and run id plus a `check` column used for the
exit code, then prints example 1's trace chain as the audit trail. The trace panel in
the app moved directly under the results, labelled "Trace a result", with the
final step selected by default.

## In-plan parameter editing

**Editing is a first-class part of the plan view.** Each step is a bordered
block with widgets generated from its `ParamSpec` (choices -> select box,
int/float -> number input bounded by min/max with step 1 / 0.01, bool ->
checkbox), prefilled with the plan's value. Plain widgets rather than a form, so
the page can tell that a change is not applied yet and disable Execute with a
reason. Widget and approval-checkbox keys include the plan version, so after an
edit or a reset every widget restarts from the plan's values and every approval
box is unticked.

**The structure is locked in the engine, with G03.** `check_edits` accepts only
`{step_id: {param: value}}` for parameters of the step's registered capability.
Anything else (a new step id, `capability`, `inputs`, `id`, an invented field, a
non-mapping) is rejected as a G03 INVALID_PARAM BLOCK whose message starts with
"Edit rejected", and nothing is applied; the plan stays blocked until a valid
edit or a reset. Values are not judged there: they are applied to a copy of the
plan and the validator judges them, so an out-of-range value is a G03 BLOCK even
though the widgets are bounded. Alternative: a new guardrail code for rejected
edits, rejected as one more code for what is, in effect, an invalid parameter.

**G14 WINDOW_UNITS_CHANGED (WARN) compares the edited plan with the planner's
proposal.** `validate_plan` takes an optional `baseline` (the proposal with
defaults filled); the engine passes it for every validation of a session. For
each step whose input frequency differs between the two dry runs, every
parameter flagged `counted_in_periods` (rolling_volatility.window,
flag_regime.min_periods) that still has the proposal's value raises G14, with a
suggestion using 5 trading days per week and 21 per month ("Consider 60 for
roughly 12 weeks"), clamped to the parameter's range. The executor and replay
validate without a baseline: G14 is a warning and cannot change the outcome.
Alternative: compare with the previous applied version, rejected because the
proposal is a fixed reference and the warning then cannot appear and disappear
depending on the order of edits.

**`resample` accepts D as a pass-through.** `freq="D"` returns the business-day
series unchanged, never `pandas.resample("D")`, which would insert weekend NaNs;
weekly or monthly data to D is still refused as upsampling. The dry run already
maps D to daily, and volatility is then annualised with 252. A user can switch
example 1 to daily by changing one parameter.

**Approvals belong to one plan version.** `plan.plan_hash` hashes the plan's
content (datasets, steps, refusal fields; not the plan id or request). Each
approval stores the hash it was given for, only approvals with the current hash
count, and any applied edit clears them anyway. The run record stores
`plan_hash`, the edit history (`edits`) and `param_diff`: for each parameter whose
final value differs from the proposal, the proposed value, the final value and
the time of the last edit. The trace marks those parameters ("edited, was 12"),
the Run history page lists them, and replay runs `plan_final`, so an edited run
replays with identical hashes.

**"Reset to planner's proposal" is an edit back to the proposal's values**, so it
goes through the same path (re-validation, approvals cleared, history recorded).

## Interface copy and polish

**Full dataset names on hover, from the catalog.** Wherever a dataset is listed
or selected, the UI shows its catalog name with the id, now in the form
"NASDAQCOM — NASDAQ Composite Index" (`ui.dataset_label`, read from
`data/catalog.json`, never hardcoded; see "Dataset names and labels" below). The sidebar multiselect and the `dataset_id` parameter select box
show the full name as their option labels and list every dataset's full name in
their help tooltip; each selected dataset's sidebar line carries the full name in
an HTML `title` attribute, so hovering the line shows it; the trace and replay
tables show the full name in a wide column. Streamlit's multiselect chips cannot
carry a per-chip tooltip, hence the help tooltip plus the hoverable lines.
Alternative: a custom component, rejected because it adds a dependency.

**Header: description, badges and "how it works", with light CSS only.** The
title is followed by a two-line description for traders and market-risk
analysts, three badges (validated, verified numbers, replayable) and the original
sentence as the "How it works" line. The CSS sets only sizes and spacing; colours
come from the active theme, so the header reads the same in light and dark mode.
No new dependency.

**Example workflows appear only in mock mode.** They are canned plans, so the
sidebar shows them only when the LLM mode is mock; in live and cached mode the
analyst types the request (a request picked in mock mode stays in the box after
switching). The "Audit trail" hint under the examples is gone; Trace a result
itself is unchanged.

**Example 3 asks for a SARIMA forecast** (`sarima_forecast_block`, replacing the
earlier model-fitting request). A forecast request is typical for a market-risk
analyst and still has no registered capability, so the outcome is unchanged: the
planner refuses (G13), and the misbehaving-planner test emits a `sarima` step
that G02 blocks. `MOCK_PROMPT_VERSION` moved to `canned-scenarios-v4`; the plan
cache key includes the request text, so plans cached for the old request are
never reused.

## Period filtering and extremes

**Where `filter_period` goes in a plan.** It usually belongs right after
`load_series` or at the end of the chain, and the planner prompt explains why the
placement changes results: filtering BEFORE a rolling or expanding calculation
shortens the history used for the window warm-up and for the expanding
threshold's baseline (a 12-week volatility "since 2020" would then only start in
March 2020, and an expanding 90th percentile would be judged against data since
2020 only),
while filtering AFTER keeps the full history for the calculation and only trims
what is reported. When the user asks to *show* a recent period, the planner filters
after the calculation, unless the user explicitly wants the calculation itself
restricted to that period. The canned "volatility since 2020" plan and its golden
case encode that default. Alternatives: always filter first (simplest to explain,
but silently changes every rolling and expanding result), or let the validator
reorder steps (rejected: the system never rewrites a plan; it only judges it).

**The resolved period is recorded.** The resolved, inclusive start and end dates
(a missing start or end becomes the first or last observation) are recorded in
`ArtifactMeta.period_start` / `period_end`, in the step record of the manifest, in
the step's notes and in the facts sheet. They are not inherited by later steps,
like notes: each artifact describes its own step.

**Start and end, at least one of them.** `start` and `end` are optional parameters
whose default is null; a new `ParamSpec.nullable` flag lets null mean "not set" for
these only (every other parameter still rejects null). `check_period` (G03) blocks
a filter with neither date, a start not before the end, and a period entirely
outside the coverage of the input's source datasets (from the catalog, so no data
is read). These reuse G03 INVALID_PARAM rather than a new
code: they are invalid parameter values, and the guardrail table stays stable.
A period that overlaps the data only partly passes; if too few rows remain, the
next step's R02 stops the run, and a new `MinObsRule.output_min_obs` makes R02 also
check `filter_period`'s own output (at least 2 rows), since a period after the last
complete week can be inside the catalog's coverage yet empty.

**A `date` parameter type and widgets for it.** Plans carry dates as "YYYY-MM-DD"
strings (JSON has no date type); the validator accepts exactly that form and only
real days (not 2021-02-30). The editor shows a date input bounded by the catalog's
date range. A nullable parameter
gets a "Set" checkbox in front of its widget, unticked meaning null, because a
Streamlit input created with a value returns that value when emptied and so can
never be cleared back to null. Edits go through `apply_param_edits` and the same
validator as every other edit.

**`find_extremes` returns a table of named rows.** `{"max": {"date", "value",
"tie_count"}, "min": {...}}`, shown as columns extreme, date, value, tie_count.
TABLE values may now be dicts of rows as well as dicts of scalars; the facts sheet
flattens rows into `max_date`, `max_value`, `max_tie_count`. Missing values are
dropped first. Ties use exact equality (the same recorded value) and report the
earliest date with the number of tied dates, so the answer is never silently
ambiguous. Alternative: `idxmax` alone, rejected because it hides ties.

**Dates are facts in the grounding check.** Before, any ISO date inside the data
range passed, so an invented peak date between the first and last observation
would have been "verified". Now a date must be a date of the facts sheet; dates
written in words ("19 November 2021", "Nov. 19, 2021") are parsed and checked the
same way, a month ("November 2021") must contain a fact date, and a year on its own
must still lie within the data range. The narrator prompt (narrator-v3) asks for
dates exactly as displayed.

**Further canned requests, kept apart from the example workflows.** Four requests
using the new capabilities have canned plans (`mock_scenarios.CANNED_REQUESTS`), matched
by exact text or id only, before the examples' keywords: "Show weekly volatility
for NASDAQ since 2020..." contains example 1's keywords but gets its own plan. They
are golden cases too, so the mock evaluation scores them, but they are not sidebar
buttons and `cli demo` still runs the five examples. The planner prompt moved to
planner-v3 (the plan cache invalidates), the canned set to canned-scenarios-v5.

**Synthetic test data now runs from 1996 to 2025.** The same 7,800 business days,
values and Monday start, shifted six years, so tests can filter periods such as
"since 2020"; weekly bars are unchanged and no existing test depended on the dates.

## Module names

**`validation.py` became `guardrails.py`, and `scenarios.py` became `mock_scenarios.py`.**
The file names now say what the modules contain: the static guardrails G01-G14
with the guardrail table, and the canned plans the mock planner answers with
(the example workflows and the further canned requests). Only the files changed
(moved with `git mv`, tests renamed to match): functions, classes and concepts
keep their names (`validate_plan`, `ValidationReport`, the "validation" stage in
monitoring, the guardrail codes). No compatibility shim was left under the old
names, so a stale import fails loudly instead of silently working.

## Dataset names and labels

**Display names checked against the official FRED titles** (fred.stlouisfed.org,
2026-10-16). SP500 is titled "S&P 500" on FRED; it is displayed as "S&P 500 Index",
with "Standard & Poor's 500" as its full name. NASDAQCOM is titled "NASDAQ
Composite" on FRED, whose notes call it the "NASDAQ Composite Index", the name kept
for display. VIXCLS and DEXSZUS match FRED exactly. DGS10 and DGS2 are titled
"Market Yield on U.S. Treasury Securities at 10-Year (2-Year) Constant Maturity,
Quoted on an Investment Basis"; the catalog held a truncated copy, and they are now
displayed as "10-Year (2-Year) U.S. Treasury Constant Maturity Yield".

**A `full_name` field next to the display `title`.** The catalog keeps the short
display name in `title` (used in the planner's dataset summary and the UI) and the
full name in `full_name` (shown on hover), so shortening a long official title loses
nothing. Dataset ids, snapshot files and their hashes are unchanged, and the
catalog's text is part of neither run records nor plan-cache keys, so existing plans,
cached plans, mock plans, golden requests and replays keep working. Requests and
canned plan rationales that say "S&P 500" are left as written: they are how
analysts phrase requests, they are matched exactly, and rationales are part of a
plan's hash.

**One label everywhere: "<FRED id> — <display name>".** `ui.dataset_label` builds it
from the catalog (e.g. "SP500 — S&P 500 Index", "DGS10 — 10-Year U.S. Treasury
Constant Maturity Yield"), and `ui.dataset_full_name` the hover text. The FRED id
is used for every dataset, so series without a market ticker look the same; it
equals the dataset id plans use, which a test checks. The label is used in the
dataset picker, the `dataset_id` parameter, the trace and replay tables, a "Source"
line under each result and a "Datasets" line in Run history.

## Relative periods

**The planner turns "the last N years" into an explicit start date.** `filter_period`
no longer has a parameter for a period relative to the data's end: a period is only
ever `start` and/or `end`. For a request such as "the last 5 years", the planner
sets `start` to the dataset's `end_date` (which it sees in the dataset metadata)
minus five years and says so in its notes. The plan the analyst reviews, edits,
approves and replays then holds the exact date, there is one way to express a
period, and nothing is resolved out of sight. The canned "last 5 years" request and
its golden case were removed; the remaining canned requests and tests use explicit
dates. Alternative: decline relative periods and ask for explicit dates (G13),
rejected as needlessly unhelpful when the dataset's last date is known.
