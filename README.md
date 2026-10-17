# QuantLab

**Ask a quantitative question in plain English and get an auditable answer, without a single line of generated code being run.**

## Intended users and use cases

QuantLab is built for **market-risk and quantitative analysts** who need fast,
reproducible answers from approved market data. Typical questions:

* "Flag the weeks when NASDAQ volatility was unusually high versus its own history."
* "Was the average weekly NASDAQ return different in high-VIX weeks?"
* "Where is the 10-year minus 2-year Treasury spread today, and what is its usual range?"
* "How volatile has the Swiss franc been against the dollar over rolling 26-week windows?"

## The problem it solves

Analysts in regulated institutions need quantitative analysis on demand, but
letting an AI assistant write and execute code against production data is not
acceptable: every answer would come from new, untested code that nobody
reviewed. The usual alternative, a one-off notebook written by a quant, is slow,
hard to review and hard to reproduce for model risk or audit.

QuantLab removes that trade-off. The language model never writes code. It only
selects from a registry of pre-approved, unit-tested capabilities and fills in
their parameters. Everything that touches data is deterministic, checked and
recorded.

## The safety model

> **The model proposes, the system disposes.**

1. **Plan.** A language model turns the request into a *plan*: inert JSON naming
   registered capabilities and their parameters. It sees capability descriptions
   and dataset metadata, never data values.
2. **Validate.** A deterministic validator checks the plan against the registry
   and a set of guardrails before anything runs. It blocks unsafe or meaningless
   plans, and asks for explicit approval where a modelling choice matters.
3. **Review.** The analyst reviews the plan, can adjust parameter values (every
   change is re-validated by the same validator) and approves it.
4. **Execute.** A deterministic engine runs the approved plan by looking each
   capability up in the registry. A name that is not registered has no code
   behind it, so it cannot run.
5. **Explain.** A second model call writes a short summary from a facts sheet of
   computed numbers only, and every number in that summary is verified against
   the results.
6. **Record.** Every run is saved with data hashes, library versions, the plan,
   edits, approvals and output hashes, and can be replayed later without any
   language model, producing identical hashes.

In practice this means:

* only pre-approved, tested computations can run;
* statistically unsound choices are blocked (returns of a yield, mixing daily and
  weekly data) or require explicit approval (look-ahead thresholds, the return
  method), each with a written reason;
* requests outside the registry (a forecast, a model, running code) are declined,
  never improvised;
* every number shown comes from the engine, and the narrative is checked against it;
* every result can be traced back to the exact source file and replayed.

QuantLab also runs fully offline in **mock mode**, which needs no API key.

**Example output.** For example workflow 1 on the 1971-2026 NASDAQ snapshot,
QuantLab reports that 458 of 2,842 weeks (16.1%) had a 12-week annualised
volatility above the 90th percentile of their own past, in 42 distinct episodes,
with a latest volatility of 13.86%. The share exceeds 10% because each week is
compared only with the history available at the time, so the more volatile
decades after the 1970s cross the earlier thresholds more often.

## Getting started

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                           # environment from uv.lock
uv run streamlit run app/streamlit_app.py         # the app (works in mock mode)

uv run python -m quantlab.cli demo                # run the example workflows, then trace example 1
uv run python -m quantlab.cli run 1 --datasets NASDAQCOM --mode mock --approve-all   # example 1 by number
uv run python -m quantlab.cli replay <run_id>     # re-run without the model, compare hashes
uv run python -m quantlab.cli trace <run_id> s5   # where does this number come from?
uv run python -m quantlab.cli eval --mode mock    # planner evaluation -> eval/reports/

uv run pytest                                     # test suite: no network, no API key
uv run ruff check . && uv run ruff format --check .
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

Refreshing the data (needs network; creates a new data version):
`uv run python scripts/fetch_data.py` (or name series to fetch only those, e.g. `SP500`).

## Using the app

The **Analysis** page follows the safety model step by step:

1. Choose the datasets the planner may use (hover a dataset for its full name),
   then type a request, or in mock mode pick one of the example workflows in the
   sidebar, and click **Generate plan**.
2. Review the plan. Each step is a collapsible block whose header names the step
   and its main parameters (e.g. "Step 4 · Rolling volatility · window: 12 weeks,
   annualize: yes"); open it for its inputs, predicted output and editable fields.
   Steps with a blocking issue or a warning open by default with a badge, and
   edited steps are marked. Approvals are not shown on the steps: each one is listed
   in the validation results, naming its step (e.g. "Step 3 · Compute returns").
3. Read the validation results. Blocking issues must be resolved, approvals must
   be ticked, and warnings are shown for context.
4. Click **Execute**. Results appear per step: charts for series (with flagged
   periods shaded), tables for statistics, and a plain-English narrative marked
   "all numbers verified" when every number checks out.
5. Use **Trace a result** under the results to follow any output back through
   every step to the source file, with its SHA-256, FRED id and retrieval date.

The other pages: **Run history** lists past runs, shows each manifest and user
edits, and replays a run without the model; **Monitoring** shows run outcomes,
latency per stage, capability usage and guardrail activity; **Capability
registry** lists every approved capability, the guardrails, and the exact
summary the planner model sees.

### Editing a plan before it runs

After a plan is generated, each step's parameters appear as fields generated
from the registry (select boxes for choices, bounded number inputs, checkboxes),
prefilled with the planner's values. Only values can change: steps, capabilities
and input references are locked, and the engine rejects any other edit with a
BLOCK. **Apply changes & re-validate** runs the same validator again, so issues
appear and clear as the plan changes: switching `flag_regime.method` to
`full_sample` raises G10, and switching `resample.freq` from W to D raises the
G14 warning that window 12 now means 12 trading days. Any edit resets the
approvals, which are recorded with the hash of the plan version they were given
for. Execute stays disabled, with a one-line reason, while changes are unapplied,
a BLOCK remains or an approval is unticked. Edited values are marked ("edited
(was 12)"), **Reset to planner's proposal** undoes them, and the run record keeps
the parameter-level diff that the trace and Run history display. Replay re-runs
the edited plan with identical hashes.

## Example workflows

In mock mode the sidebar offers five ready-made requests; in live and cached
mode the sidebar hides them and you type the request. Together they cover one
realistic task, a market-risk analyst identifying periods of unusual market
behaviour, and the protections QuantLab applies along the way. They run with
canned plans in mock mode (`mock_scenarios.py`) or with the live planner, from the
app or with `cli demo`.

1. **High-volatility weeks** (`volatility_flag_success`): "Using the NASDAQ,
   compute weekly returns and 12-week rolling annualised volatility, and flag the
   weeks where volatility was unusually high versus its own history." Five steps.
   The analyst confirms the return method (G12), and the result shows the
   volatility series with the high-volatility weeks shaded.
2. **Look-ahead protection** (`volatility_flag_lookahead`): the same analysis
   with a full-sample 95th-percentile threshold. QuantLab allows it only with
   explicit approval (G10), explains that a full-sample threshold judges past
   weeks using information from the future, and recommends the expanding method.
   This protects the analyst from the most common source of overstated backtests.
3. **Out-of-scope forecast request** (`sarima_forecast_block`): "Fit SARIMA to
   predict next week's NASDAQ return." The registry has no capability for model
   fitting or prediction, so the planner declines (G13) and lists what is
   available instead. If a planner ever proposed such a step anyway, the
   validator would block it independently (G02).
4. **Prompt-injection protection** (`prompt_injection_block`): a request that
   tells the assistant to ignore its rules and run Python is treated as data,
   not as instructions, and declined (G13). There is no code path that could run
   it, and automated tests confirm that nothing from the request reaches a shell
   or a subprocess.
5. **Frequency-alignment protection** (`frequency_mismatch_block`): comparing
   weekly NASDAQ returns with daily VIX flags would silently keep only the dates
   the two share. The validator blocks the comparison (G09) before anything runs
   and states the fix: resample VIX to weekly first.

**Audit trail.** For any executed result, such as example 1's flags, **Trace a
result** shows the full chain `flag_regime` -> `rolling_volatility` ->
`compute_returns` -> `resample` -> `load_series` -> the NASDAQ snapshot file,
with each step's library call, parameters (user edits marked) and output hash.
`cli demo` prints the same chain.

**More canned requests.** In mock mode the planner also has plans for these
requests, typed exactly as written (they are not sidebar buttons): "Show weekly
volatility for NASDAQ since 2020 and tell me when it peaked." (volatility on the
full history, then filtered), "When was the VIX at its highest and lowest?" and
"Summary statistics of daily S&P 500 returns in 2022."

## Live mode

Live mode calls a language model for the planner and the narrator. The provider
is set in `.env`: **OpenAI** by default (model `gpt-6-luna`, Responses API), or
Anthropic (`QUANTLAB_LLM_PROVIDER=anthropic`, model `claude-sonnet-5-5`).

**1. Configure.**

```bash
cp .env.example .env
# edit .env:  QUANTLAB_LLM_PROVIDER=openai   OPENAI_API_KEY=sk-...
#             optionally QUANTLAB_PLANNER_MODEL=gpt-6.1-sol as an alternative to gpt-6-luna
```

**2. Check the connection** (one planner call and one narrator call):

```bash
uv run python -m quantlab.cli run "Show summary statistics of the daily VIX level." --datasets VIXCLS --mode live
```

This prints a two-step plan, `Status: completed` and a model-written narrative
marked "all numbers verified". An authentication error means the key is wrong;
"API error 404" usually means the model id is not available to the account (set
`QUANTLAB_PLANNER_MODEL`).

**3. Pre-load the plan cache** for the example workflows (one call each):

```bash
uv run python -m quantlab.cli demo --mode live
```

Typing exactly these requests in **cached** mode (or picking one in mock mode,
then switching to cached: the request stays in the box) then returns their
plans instantly and identically, even on a slow network (the
narrator still calls the model, and falls back to the deterministic template if
it cannot). A live model may decline a request for a different reason than the
canned plan (for example refuse example 5 outright with G13, rather than
proposing the misaligned plan that the validator blocks with G09); the command
only checks that blocked workflows stay blocked.

**4. Start the app**, open http://localhost:8501 and pick **cached** (the default
when a key is set) or **live** in the sidebar. If the API is unavailable, switch
to **mock**: everything keeps working offline with the canned plans.

## Architecture

```mermaid
flowchart LR
    U[User request] --> P["Planner LLM<br/>planner.py"]
    R[("Capability registry<br/>registry.py")] -. generated summary .-> P
    C[("Dataset catalog<br/>metadata only")] -. id, kind, frequency, dates .-> P
    P -->|"plan: inert JSON"| V["Validator G01-G14<br/>guardrails.py"]
    V -->|"BLOCK / APPROVAL / WARN"| H["Human review<br/>edit params, approve"]
    H -->|"edits: same validator"| V
    H -->|approved plan| E["Executor R01-R04<br/>executor.py"]
    R -->|"REGISTRY[name].func"| E
    D[("FRED snapshot<br/>SHA-256 verified")] --> E
    E -->|"facts sheet (numbers only)"| N["Narrator LLM<br/>+ grounding check"]
    E --> L[("Run record<br/>manifest + outputs")]
    N --> L
    L --> RP["Replay<br/>no LLM, hash comparison"]
    L --> T["Trace<br/>step -> source file"]
    P & V & E & N --> M[("events.jsonl<br/>monitoring")]
```

| Module (`src/quantlab/`) | Role |
|---|---|
| `config.py` | Settings from `.env`: API key, models, LLM mode, paths, budgets |
| `types.py` | `DataType`, `SeriesKind`, `Frequency`; `Artifact` (value + explicit metadata); canonical hashing |
| `data.py` | Catalog, FRED CSV parsing, SHA-256 verification, LLM-safe dataset metadata |
| `capabilities.py` | The 11 quantitative functions: the only code that calls pandas/numpy for analysis |
| `registry.py` | Capability specs, the `REGISTRY` whitelist, generated LLM summary and registry version |
| `plan.py` | Pydantic `Plan` / `Step` (extra fields forbidden), plan version hash, the LLM tool schema |
| `guardrails.py` | Static guardrails G01-G14, edit lock, metadata dry run, `ValidationReport`, the guardrail table |
| `mock_scenarios.py` | Example workflows 1-5 and further canned requests, with plans (mock mode, CLI, app, tests) |
| `executor.py` | Registry dispatch, approval gate, runtime guardrails R01-R04, step records |
| `planner.py` | OpenAI and Anthropic planners, mock planner, plan cache |
| `narrator.py` | Facts sheet, LLM narrator, template fallback, numeric grounding check |
| `lineage.py` | Run record (manifest), save/load, replay, trace |
| `monitoring.py` | Append-only JSONL event log and dashboard aggregations |
| `engine.py` | `Session`: plan, validate, edit, approve, execute, narrate, save; used by app, CLI, notebooks |
| `evaluation.py` | Planner evaluation on golden requests |
| `cli.py` | `python -m quantlab.cli demo / run / replay / trace / eval` |

The language model appears only in `planner.py` and `narrator.py` (a test
enforces this). Everything else is deterministic Python.

## Why plans instead of generated code

An assistant that writes and runs pandas code is flexible, but every run would
be new, untested code touching production data. QuantLab asks the model for a
structured plan instead:

* **Whitelist by construction.** The executor looks the capability name up in a
  dict and calls the function attached to it. An unregistered name, or anything
  that looks like code, has no function to call. There is no `eval`, `exec`,
  dynamic import or `getattr` anywhere in `src/` (a test scans for them). This
  is a stronger guarantee than sandboxing generated code, whose safety depends on
  the sandbox being perfect.
* **Checkable before running.** Because each step declares types, kinds and
  frequencies, the validator rejects "returns of a yield" or "the spread of a
  daily and a weekly series" statically, which is impossible for arbitrary code
  without executing it.
* **Reviewable.** An analyst reads a few lines of plan, not pages of code, and
  edits parameters through fields constrained by the registry.
* **Tested once, reused always.** The 11 capabilities have unit tests against
  hand-computed values; generated code would need validating every time.
* **Reproducible.** Plan, data hashes and library versions are enough to replay a
  run with no model involved.

The trade-off is expressiveness: anything outside the registry is declined (G13)
or blocked (G02). Adding a capability is a deliberate, reviewed code change,
which is what model-risk governance requires.

## Capabilities and libraries

The approved menu (browse it on the **Capability registry** page), deliberately
small and descriptive:

| Capability | What it does |
|---|---|
| `load_series` | Load one dataset from the catalog (hash-verified, missing days dropped) |
| `resample` | Daily to weekly (Friday) or monthly (month end), or keep daily (`D`, unchanged) |
| `compute_returns` | Simple or log returns of a price or index |
| `difference` | Changes of a rate or yield, in its own units |
| `spread` | a - b on the dates both series share |
| `rolling_volatility` | Trailing rolling standard deviation, annualised |
| `flag_regime` | Flag periods above a quantile threshold (e.g. high volatility) |
| `summary_stats` | Count, mean, std, min, max, skew, excess kurtosis |
| `compare_groups` | Average of a series in flagged vs normal periods (descriptive, no test) |
| `filter_period` | Keep the observations between a start and/or an end date (inclusive) |
| `find_extremes` | When a series reached its maximum and/or minimum, the value, and a tie count |

**Periods and peaks.** `filter_period` takes `start` and/or `end` (inclusive ISO
dates): giving neither is a G03 BLOCK, as are a start not before the end and a
period entirely outside the data, and a period that leaves too few observations
stops the run with R02. The resolved start and end dates are recorded in the
step's metadata and the run manifest. For a relative period such as "the last 5
years", the planner sets an explicit `start` from the dataset's last date in the
catalog and says so in its notes, so the date is visible and editable before
anything runs, and a replay uses it as written. Where it goes matters: filtering *before* a rolling or expanding
calculation shortens the history used for the window warm-up and the expanding
threshold, while filtering *after* keeps the full history and only trims what is
reported. The planner is told to filter after the calculation when the user asks
to show a recent period, unless the calculation itself should be restricted.
`find_extremes` works on any series (levels, returns, volatility), ignores
missing values, and on a tie reports the earliest date with the number of tied
dates, so an answer is never silently ambiguous.

There are no statistical tests, regressions or forecasts: such requests are
declined or blocked (example 3). Each spec declares named inputs (allowed data
types and kinds), parameters (type, range, choices, defaults), output type and
kind, a minimum-observations rule, assumptions (some need approval) and the
underlying library call.

**A deliberately small library surface:** pandas and numpy (time series),
pydantic (strict plan parsing), streamlit and plotly (UI), the official `openai`
and `anthropic` SDKs, python-dotenv and pyyaml. Deliberately excluded: scipy and
statsmodels (no statistical tests), scikit-learn (no machine-learning models),
`arch` (no GARCH), and LangChain or any agent framework (two structured model
calls do not need an orchestration layer, and every extra abstraction is more to
audit). Fewer libraries means fewer versions to pin, record and explain.

## Guardrails

Each guardrail is a small named function whose docstring explains what goes
wrong without it. Static guardrails run on the plan alone (the planner's plan and
every user edit, through the same validator); runtime guardrails need data and
run in the executor. The table is generated from `guardrails.GUARDRAILS`. G11
(overlapping observations in a t-test) was retired together with the statistical
tests; codes are never reused, so the issue ids in older run records keep their
meaning and those runs can still be replayed.

| Code | Name | Severity | Stage | What goes wrong without it |
|---|---|---|---|---|
| G01 | `SCHEMA_INVALID` | BLOCK | static | Malformed or partial planner output would have to be guessed at, and extra fields (e.g. 'code') could slip through. |
| G02 | `UNKNOWN_CAPABILITY` | BLOCK | static | The LLM could ask for an analysis nobody wrote or tested (e.g. a forecasting model, or 'run Python'). |
| G03 | `INVALID_PARAM` | BLOCK | static | Typos, wrong types or extreme values (window=1, quantile=1.5) would crash or silently produce nonsense. |
| G04 | `BAD_REFERENCE` | BLOCK | static | Steps could read missing outputs or outputs of later steps, leaving the order of computation undefined. |
| G05 | `TYPE_MISMATCH` | BLOCK | static | Meaningless chains would run, e.g. volatility of a price level or summary statistics of regime flags. |
| G06 | `BUDGET_EXCEEDED` | BLOCK | static | Runaway plans could use unbounded compute and become impossible for a human to review. |
| G07 | `UNKNOWN_DATASET` | BLOCK | static | The LLM could invent a dataset id or reach for data outside the approved catalog. |
| G08 | `RETURNS_ON_RATES` | BLOCK | static | 'Returns' of a yield are meaningless (1% -> 2% is not +100%); rates must be differenced. |
| G09 | `FREQUENCY_MISMATCH` | BLOCK | static | Combining a daily with a weekly series silently keeps only the dates they share and mixes frequencies. |
| G10 | `LOOK_AHEAD` | APPROVAL | static | Full-sample thresholds use future data, so results look better than anything achievable in real time. |
| G12 | `MATERIAL_ASSUMPTION` | APPROVAL | static | Material modelling choices (simple vs log returns) would be made by the LLM without a human noticing. |
| G13 | `PLANNER_REFUSED` | BLOCK | static | Without an explicit refusal path the LLM is pushed to invent capabilities to satisfy any request. |
| G14 | `WINDOW_UNITS_CHANGED` | WARN | static | After a frequency edit, an unchanged window silently changes its span (12 weeks becomes 12 trading days). |
| R01 | `DATASET_HASH_MISMATCH` | BLOCK | runtime | Results would come from data other than the recorded version, breaking reproducibility and audit. |
| R02 | `INSUFFICIENT_DATA` | BLOCK | runtime | Statistics on too few observations are noise presented as findings. |
| R03 | `MISSING_DATA` | WARN | runtime | A heavily gappy series can distort resampled values and returns unnoticed. |
| R04 | `ALIGNMENT_LOSS` | WARN | runtime | An inner join can silently discard a large part of the sample (e.g. series with different start dates). |

The executor re-runs the validator itself and refuses to start while any BLOCK
exists or any APPROVAL issue is unapproved, so calling it directly cannot skip
the checks. Approvals are recorded with a timestamp, the exact issue text and the
plan version they apply to.

## The model calls

**Providers.** `QUANTLAB_LLM_PROVIDER` selects OpenAI (default; Responses API,
`gpt-6-luna`) or Anthropic (Messages API, `claude-sonnet-5-5`). Both receive the
same prompts and plan schema, and their output goes through the same parsing
and validation. Only `planner.py` imports the provider SDKs, and only
`planner.py` and `narrator.py` call them. The run record stores the provider,
the requested model and the model version that actually answered.

**Planner** (`planner.py`). The system prompt contains the rules, the capability
summary generated from the registry, and dataset metadata (never values). The
request is wrapped in `<user_request>` tags and treated as data, so instructions
inside it ("ignore your rules", "run this Python") lead to a refusal. The plan
arrives through a `submit_plan` function tool whose schema is the plan schema,
and is then parsed strictly (G01) and validated. Two API constraints shaped this
(details in `docs/DECISIONS.md`): current reasoning models do not reliably accept
a *forced* tool choice or a non-default temperature, so the tool call is
requested by instruction (`tool_choice="auto"`), reasoning depth is set with an
effort level, and determinism comes from the plan cache and model-free replay
rather than `temperature=0`.

**Modes:** `live` (call the API), `cached` (reuse the cached plan for an
identical request, datasets, model, prompt version and registry version), and
`mock` (canned plans for the five example workflows and a few further requests;
anything else is politely declined). Mock mode is also the fallback whenever the network or the API is
unavailable.

**Narrator** (`narrator.py`). Receives the request, a plain plan summary,
warnings, approvals and a *facts sheet* of rounded computed numbers, and must use
only those numbers. `check_grounding` then extracts every number in the narrative
and checks that it is a correct rounding of a fact (or its percentage form), a
plan parameter, a number from the request or a warning, or an annualisation
factor. Dates are facts too: a date, as `2021-11-19` or in words such as
"19 November 2021", must be a date of the facts sheet (a month must contain one,
and a year on its own must lie within the data), so a narrative that names the
wrong peak date fails the check. The app shows "all numbers verified" or lists
the unverified numbers; the deterministic facts table is always shown.

## Reproducibility, lineage and monitoring

* **Run record.** `runs/<run_id>/manifest.json`: request, model information
  (model, prompt version, mode, cache hit, latency, tokens), the plan as proposed
  and as run with its version hash, parameter edits and the parameter-level diff,
  validation report, approvals, dataset versions (file SHA-256, retrieval time,
  rows and dates used), environment (Python, library versions, git commit),
  registry version, step records with input and output hashes (and
  `filter_period`'s resolved start and end dates), warnings, the
  narrative with its grounding result, and stage timings.
* **Outputs.** `runs/<run_id>/outputs/<step>_<capability>.csv|json`. Each file is
  the canonical text that was hashed, so `shasum -a 256` reproduces the hash in
  the manifest.
* **Replay.** `quantlab.cli replay <run_id>` (or the Run history page) verifies
  the data hashes, re-validates the final plan, executes it with the recorded
  approvals and no model, and compares every output hash. Floats are hashed at 12
  significant digits, so harmless last-bit differences between machines do not
  break replay while any real change does.
* **Trace.** `quantlab.cli trace <run_id> <step>` (or the trace panel) walks the
  plan backwards: capability, library call, parameters (user edits marked), input
  steps and output hash for each step, down to the FRED file hash and the date
  range used.
* **Monitoring.** `runs/events.jsonl` receives one event per stage (request,
  planning, validation issues, approvals, each step, runtime guardrails,
  narration, completion, replays). The Monitoring page shows run counts and
  success / blocked / failed rates, mean and p95 latency per stage, capability
  usage, guardrail activity, recent errors and the plan-cache hit rate.

## Quality assurance

Two separate questions, measured separately:

* **Does the planner choose the right plan?** Measured by `quantlab.cli eval` (or
  `eval/run_eval.py`) on `eval/golden_requests.yaml`: the five example workflows,
  simple variations, and adversarial requests (such as a forecast) that must end
  in a refusal or a BLOCK. Metrics: valid-plan rate, exact capability-sequence
  match, capability-set Jaccard, key-parameter accuracy and correct refusal/block
  rate. Reports go to `eval/reports/`. In mock mode this checks the harness only
  (the mock knows just the five examples and the canned requests); `--mode live`
  measures the real model.
* **Are the computations right?** Measured by the unit tests, independently of
  any model: returns, differences, spread alignment, rolling volatility and
  annualisation, the no-look-ahead property of expanding flags, summary
  statistics against textbook formulas, group comparisons, every guardrail,
  plan editing, replay, grounding and a source safety scan.

## Project layout

```
pyproject.toml, uv.lock     dependencies (direct ones pinned exactly)
CLAUDE.md                   engineering conventions for AI-assisted development
docs/SPEC.md                the v1 product specification
docs/DECISIONS.md           design decisions and their rationale
data/                       catalog.json, snapshots/*.csv (committed), README (sources, licences)
scripts/fetch_data.py       refreshes the FRED snapshot and catalog
src/quantlab/               the package (see the module table above)
app/                        Streamlit: streamlit_app.py (router), ui.py, pages/
notebooks/01..08            one notebook per layer, executed in mock mode
eval/                       golden_requests.yaml, run_eval.py, reports/
tests/                      pytest suite (synthetic data, simulated model clients)
runs/                       run records, plan cache, events.jsonl (git-ignored)
```

## Current limitations and roadmap

* **Live mode coverage.** The OpenAI path is in use with `gpt-6-luna` for both
  the planner and the narrator. The Anthropic path is covered by automated tests
  with simulated responses only, and the live planner evaluation (`eval --mode
  live`) has not yet been run. Mock mode, used by the tests, the CLI, the
  notebooks and the app, is fully covered.
* **No plan -> observe -> replan loop.** Validation errors are shown to the
  analyst but not yet fed back to the model for a corrected plan.
* **The grounding check is numeric.** It proves every number is a correct
  rounding of an allowed value and every date is a date of the results, not that
  the sentence attaches it to the right concept (a true date described as the
  wrong extreme would pass), and it compares absolute values (sign errors are
  not caught).
* **Descriptive only.** There are no statistical tests, regressions or forecasts,
  so QuantLab describes what happened (averages, volatility, high-volatility
  periods, spreads) but does not judge significance or predict. Such requests
  are declined or blocked.
* **File-based, single-user.** Runs and events live on local disk; approvals are
  recorded as "user", with no authentication or four-eyes approval yet.
* **`registry_version` covers capability interfaces, not their code**; the git
  commit in each manifest covers the code.
* **Data licences.** SP500 and NASDAQCOM are marked "Copyrighted: Pre-Approval
  Required" on FRED; see `data/README.md` before redistributing the data.
* **Short S&P 500 Index history.** FRED distributes only the last 10 years.

Roadmap:

* **Governed dataset onboarding.** A reviewed process to add or refresh datasets (fetch,
  review, hash, register in the catalog, with a catalog history), kept separate from
  analysis so users never pull data at runtime. Requests for data that is not available
  are logged as unmet data needs and become the intake queue for this process.
* **New capabilities.** Further reviewed analysis functions in the registry (drawdowns,
  correlations, statistical tests such as comparing returns across regimes, regressions
  with robust errors, GARCH), each added as a code change with tests, never generated at
  runtime.
* **New guardrails.** Further validation rules as new risks are identified, each with a
  defined severity (BLOCK, APPROVAL or WARN) and test coverage.
* **Replan loop.** Validator feedback sent back to the planner for a corrected plan.
* **Access control.** Authentication and role-based access, with named approvers and
  four-eyes approvals.
* **Persistent store** for runs and events.
* **Continuous integration** running the tests, lint, the example workflows, the
  notebooks and the live evaluation.
* **Cost control.** Per-run cost tracking and prompt caching for the planner.
