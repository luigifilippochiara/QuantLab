# QuantLab

QuantLab turns natural-language requests into quantitative analyses, in a
setting where the AI may only use **approved internal capabilities** and must
**never generate or execute code**.

> **The LLM proposes, the system disposes.** An LLM writes a *plan*: inert JSON
> naming registered capabilities and their parameters. A deterministic validator
> checks it, a human reviews and approves it, a deterministic engine runs it, and
> a second LLM call summarises the results using only numbers the engine
> computed, which are then verified one by one. Every run is recorded and can be
> replayed without any LLM, with identical output hashes.

Everything works offline in **mock mode** (no API key needed).

---

## Business use case

A markets or risk team wants quick, *auditable* answers to questions like:

> "Do NASDAQ returns over the following month differ after high-volatility weeks?"
> "Is the 10y-2y Treasury spread mean-reverting?"

Today that means a quant writing a one-off notebook: slow, hard to review, and
hard to reproduce for model-risk or audit. QuantLab gives self-service access to
a small menu of vetted analyses while keeping the controls a bank needs:

* only pre-approved, unit-tested computations can run (no generated code);
* statistically dangerous choices are blocked (returns of a yield, mixed
  frequencies) or need explicit human approval (look-ahead thresholds, return
  method), with a written reason;
* every number shown to the user comes from the deterministic engine, and the
  narrative is checked against those numbers;
* every run leaves a manifest (data hashes, library versions, plan, edits,
  approvals, output hashes) that can be replayed and traced back to the source file.

For example, on the 1971-2026 NASDAQ snapshot (scenario A), 4-week returns after
high-volatility weeks average 1.446% versus 0.839% after normal weeks (non-overlapping
sample), but the difference is not statistically significant (Welch p = 0.5009;
Newey-West regression on all overlapping observations, p = 0.4966).

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                           # environment from uv.lock
uv run pytest                                     # full test suite: no network, no API key
uv run ruff check .                               # lint (also: uv run ruff format --check .)

uv run python -m quantlab.cli demo                # all 6 demo scenarios, mock mode
uv run python -m quantlab.cli replay <run_id>     # re-run without the LLM, compare hashes
uv run python -m quantlab.cli trace <run_id> s7   # where does this number come from?
uv run python -m quantlab.cli run "Weekly 10y minus 2y Treasury spread, summary statistics, 4-week changes, and an ADF stationarity test on the spread." --datasets DGS10 DGS2 --mode mock
uv run python -m quantlab.cli eval --mode mock    # planner evaluation -> eval/reports/

uv run streamlit run app/streamlit_app.py         # the app (works in mock mode)

uv run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

Refreshing the data (needs network; creates a new data version):
`uv run python scripts/fetch_data.py`.

## Running the live demo

Live mode calls a real LLM for the planner and the narrator. The provider is
set in `.env`: **OpenAI** by default (model `gpt-6.1-sol`, Responses API), or
Anthropic (`QUANTLAB_LLM_PROVIDER=anthropic`, model `claude-sonnet-5-5`).

**1. Configure (once).**

```bash
cp .env.example .env
# edit .env:  QUANTLAB_LLM_PROVIDER=openai   OPENAI_API_KEY=sk-...
#             optionally QUANTLAB_PLANNER_MODEL=gpt-6-luna (cheaper) or gpt-6-astra (strongest)
```

**2. Smoke-test the key and model** (one planner call and one narrator call):

```bash
uv run python -m quantlab.cli run "Show summary statistics of the daily VIX level." --datasets VIXCLS --mode live
```

It should print a two-step plan, `Status: completed` and a narrative written by
the LLM with "all numbers verified". An authentication error means the key is
wrong; "API error 404" usually means the model id is not available to your
account (set `QUANTLAB_PLANNER_MODEL`).

**3. Warm the plan cache before presenting** (one LLM call per scenario):

```bash
uv run python -m quantlab.cli demo --mode live
```

The scenario buttons in the app send exactly these requests, so in **cached**
mode the plans then appear instantly and identically, even if the network is
slow during the interview (the narrator still calls the LLM, and falls back to
the deterministic template if it cannot). A real LLM may block a scenario for a
different reason than the mock (for example refuse C outright instead of
proposing a random forest the validator blocks), which is fine: the demo
command only checks that blocked scenarios stay blocked.

**4. Start the app** and open http://localhost:8501:

```bash
uv run streamlit run app/streamlit_app.py
```

In the sidebar, pick **cached** (the default when a key is set) or **live**,
then a scenario button, **Generate plan**, review, approve, **Execute**. If the
API misbehaves, switch the sidebar to **mock**: everything keeps working
offline with the canned plans.

A 15-minute path through the app:

1. **Capability registry** page: the approved menu, the guardrail table, and
   the exact capability summary the LLM sees.
2. **Analysis, scenario A**: the plan table and raw JSON, the G12 approval that
   gates Execute, the volatility chart with shaded high-volatility regimes, the
   two test cards, the narrative with its "all numbers verified" badge, the
   facts sheet, then **Trace** step s7 back to the NASDAQ file hash. Optionally
   edit a parameter (e.g. window 26) to show it is re-validated and the approval
   is asked again.
3. **Scenario B**: look-ahead (G10) and material-assumption (G12) approvals plus
   the overlapping-observations warning (G11).
4. **Scenarios C and D**: an unregistered capability and returns of a yield are
   blocked; the prompt injection is refused. No code is ever run.
5. **Scenario E**: daily VIX flags against weekly returns, blocked by G09.
6. **Scenario F**: the rates analysis, with an honest R04 warning (the 2-year
   yield starts 14 years after the 10-year).
7. **Run history**: open scenario A's manifest and **Replay** it without the
   LLM (every output hash matches). **Monitoring**: runs, block rate, latency
   per stage (planning is where the LLM time goes), guardrail activity.

## Architecture

```mermaid
flowchart LR
    U[User request] --> P["Planner LLM<br/>planner.py"]
    R[("Capability registry<br/>registry.py")] -. generated summary .-> P
    C[("Dataset catalog<br/>metadata only")] -. id, kind, frequency, dates .-> P
    P -->|"plan: inert JSON"| V["Validator G01-G13<br/>validation.py"]
    V -->|"BLOCK / APPROVAL / WARN"| H["Human review<br/>edit params, approve"]
    H -->|"edits: same validator"| V
    H -->|approved plan| E["Executor R01-R05<br/>executor.py"]
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
| `capabilities.py` | The 12 quantitative functions: the only code that calls pandas/scipy/statsmodels for analysis |
| `registry.py` | Capability specs, the `REGISTRY` whitelist, generated LLM summary and registry version |
| `plan.py` | Pydantic `Plan` / `Step` (extra fields forbidden) and the LLM tool schema |
| `validation.py` | Static guardrails G01-G13, metadata dry run, `ValidationReport`, the guardrail table |
| `scenarios.py` | Demo scenarios A-F with canned plans (used by mock mode, CLI, app, tests) |
| `executor.py` | Registry dispatch, approval gate, runtime guardrails R01-R05, step records |
| `planner.py` | OpenAI and Anthropic planners, mock planner, plan cache |
| `narrator.py` | Facts sheet, LLM narrator, template fallback, numeric grounding check |
| `lineage.py` | Run record (manifest), save/load, replay, trace |
| `monitoring.py` | Append-only JSONL event log and dashboard aggregations |
| `engine.py` | `Session`: plan, validate, edit, approve, execute, narrate, save; used by app, CLI, notebooks |
| `evaluation.py` | Planner evaluation on golden requests |
| `cli.py` | `python -m quantlab.cli demo / run / replay / trace / eval` |

The LLM appears only in `planner.py` and `narrator.py` (a test enforces this).
Everything else is deterministic Python.

## Why plan-as-data instead of code generation

An assistant that writes and runs pandas code is flexible, but in a bank every
run would be new, untested code touching production data. QuantLab instead
makes the LLM fill in a structured plan:

* **Whitelist by construction.** The executor looks the capability name up in a
  dict and calls the function attached to it. An unregistered name, or anything
  that looks like code, has no function to call. There is no `eval`, `exec`,
  dynamic import or `getattr` anywhere in `src/` (a test scans for them). Compare
  this with sandboxing generated code, where safety depends on the sandbox being perfect.
* **Checkable before running.** Because each step declares types, kinds and
  frequencies, the validator can reject "returns of a yield" or "daily flags
  against weekly returns" statically, which is impossible for arbitrary code
  without executing it.
* **Reviewable.** A human reads eight lines of plan, not eighty lines of code,
  and edits parameters through widgets constrained by the registry.
* **Tested once, reused always.** The 12 capabilities have unit tests against
  hand-computed values; generated code would need validating every time.
* **Reproducible.** Plan + data hashes + library versions are enough to replay a
  run with no LLM involved.

The trade-off is expressiveness: anything outside the registry is refused (G13)
or blocked (G02). Adding a capability is a deliberate, reviewed code change,
which is how model risk governance wants it.

## Capabilities and libraries

The approved menu (browse it on the **Capability registry** page):

`load_series`, `resample`, `compute_returns`, `difference`, `spread`,
`rolling_volatility`, `flag_regime`, `forward_return`, `summary_stats`,
`compare_groups` (Welch t / Mann-Whitney), `ols_hac` (Newey-West), `adf_test`.

Each spec declares named inputs (allowed data types and kinds), parameters
(type, range, choices, defaults), output type and kind, a minimum-observations
rule, assumptions (some need approval) and the underlying library call.

**Library surface, deliberately small:** pandas and numpy (time series), scipy
(Welch and Mann-Whitney tests), statsmodels (OLS with HAC errors, ADF), pydantic
(strict plan parsing), streamlit and plotly (UI), the official `openai` and
`anthropic` SDKs, python-dotenv and pyyaml. Deliberately **not** included: scikit-learn (no ML in
v1; "fit a random forest" is a test case for refusal), `arch` (GARCH is a later
capability; rolling volatility suffices for regime flags), and LangChain or any
agent framework (two structured LLM calls do not need an orchestration layer,
and every extra abstraction is more to audit). Fewer libraries means fewer
versions to pin, record and explain.

## Guardrails

Each guardrail is a small named function whose docstring explains what goes
wrong without it. Static guardrails run on the plan alone (planner output and
every user edit, through the same validator); runtime guardrails need data and
run in the executor. The table is generated from `validation.GUARDRAILS`.

| Code | Name | Severity | Stage | What goes wrong without it |
|---|---|---|---|---|
| G01 | `SCHEMA_INVALID` | BLOCK | static | Malformed or partial planner output would have to be guessed at, and extra fields (e.g. 'code') could slip through. |
| G02 | `UNKNOWN_CAPABILITY` | BLOCK | static | The LLM could ask for an analysis nobody wrote or tested (e.g. a random forest, or 'run Python'). |
| G03 | `INVALID_PARAM` | BLOCK | static | Typos, wrong types or extreme values (window=1, quantile=1.5) would crash or silently produce nonsense. |
| G04 | `BAD_REFERENCE` | BLOCK | static | Steps could read missing outputs or outputs of later steps, leaving the order of computation undefined. |
| G05 | `TYPE_MISMATCH` | BLOCK | static | Meaningless chains would run, e.g. volatility of a price level or regime flags built on forward returns. |
| G06 | `BUDGET_EXCEEDED` | BLOCK | static | Runaway plans could use unbounded compute and become impossible for a human to review. |
| G07 | `UNKNOWN_DATASET` | BLOCK | static | The LLM could invent a dataset id or reach for data outside the approved catalog. |
| G08 | `RETURNS_ON_RATES` | BLOCK | static | 'Returns' of a yield are meaningless (1% -> 2% is not +100%); rates must be differenced. |
| G09 | `FREQUENCY_MISMATCH` | BLOCK | static | Joining daily with weekly series silently keeps only overlapping dates and mixes horizons. |
| G10 | `LOOK_AHEAD` | APPROVAL | static | Full-sample thresholds use future data, so results look better than anything achievable in real time. |
| G11 | `OVERLAPPING_OBSERVATIONS` | WARN | static | Overlapping multi-period returns are autocorrelated; a plain t-test then overstates significance. |
| G12 | `MATERIAL_ASSUMPTION` | APPROVAL | static | Material modelling choices (simple vs log returns) would be made by the LLM without a human noticing. |
| G13 | `PLANNER_REFUSED` | BLOCK | static | Without an explicit refusal path the LLM is pushed to invent capabilities to satisfy any request. |
| R01 | `DATASET_HASH_MISMATCH` | BLOCK | runtime | Results would come from data other than the recorded version, breaking reproducibility and audit. |
| R02 | `INSUFFICIENT_DATA` | BLOCK | runtime | Statistics on too few observations are noise presented as findings. |
| R03 | `MISSING_DATA` | WARN | runtime | A heavily gappy series can distort resampled values and returns unnoticed. |
| R04 | `ALIGNMENT_LOSS` | WARN | runtime | An inner join can silently discard a large part of the sample (e.g. series with different start dates). |
| R05 | `LOW_EFFECTIVE_SAMPLE` | WARN | runtime | Many flagged weeks from a handful of episodes look like a big sample but carry little independent information. |

The executor re-runs the validator itself and refuses to start while any BLOCK
exists or any APPROVAL issue is unapproved, so calling it directly cannot skip
the checks. Approvals are recorded with a timestamp and the exact issue text.

## The LLM calls

**Providers.** `QUANTLAB_LLM_PROVIDER` selects OpenAI (default; Responses API,
`gpt-6.1-sol`) or Anthropic (Messages API, `claude-sonnet-5-5`). Both receive the
same prompts and plan schema, and their output goes through the same parsing
and validation; only `planner.py` imports the provider SDKs, and only
`planner.py` and `narrator.py` call them. The run record stores the provider,
the requested model and the model version that actually answered.

**Planner** (`planner.py`). The system prompt contains rules, the capability
summary generated from the registry, and dataset metadata (never values). The
request is wrapped in `<user_request>` tags and treated as data: instructions
inside it ("ignore your rules", "run this Python") lead to a refusal. Output
arrives through a `submit_plan` function tool whose schema is the plan schema,
and is then parsed strictly (G01) and validated. Two API constraints shaped this
(details in `docs/DECISIONS.md`): current reasoning models do not reliably accept
a *forced* tool choice or a non-default temperature, so the tool call is
requested by instruction (`tool_choice="auto"`), reasoning depth is set with an
effort level, and determinism comes from the plan cache and LLM-free replay
rather than `temperature=0`.

**Modes:** `live` (call the API), `cached` (reuse the cached plan for an
identical request, datasets, model, prompt version and registry version), and
`mock` (canned plans for the six demo scenarios; anything else is politely
refused). Mock mode is also the fallback if the network fails during a demo.

**Narrator** (`narrator.py`). Gets the request, a plain plan summary, warnings,
approvals and a *facts sheet* of rounded computed numbers. It must use only
those numbers. `check_grounding` then extracts every number in the narrative
and checks it is a correct rounding of a fact (or its percentage form), a plan
parameter, a number from the request or a warning, a conventional significance
level, or a date in the data range. The app shows "all numbers verified" or the
list of unverified numbers; the deterministic facts table is always shown.

## Reproducibility, lineage and monitoring

* **Run record.** `runs/<run_id>/manifest.json`: request, LLM info (model,
  prompt version, mode, cache hit, latency, tokens), the plan as proposed and as
  run, parameter edits, validation report, approvals, dataset versions (file
  SHA-256, retrieval time, rows and dates used), environment (Python, library
  versions, git commit), registry version, step records with input and output
  hashes, warnings, narrative with its grounding result, stage timings.
* **Outputs.** `runs/<run_id>/outputs/<step>_<capability>.csv|json`. Each file is
  the canonical text that was hashed, so `shasum -a 256` reproduces the hash in
  the manifest.
* **Replay.** `quantlab.cli replay <run_id>` (or the Run history page) verifies
  the data hashes, re-validates the final plan, executes it with the recorded
  approvals and no LLM, and compares every output hash. Floats are hashed at 12
  significant digits, so harmless last-bit differences between machines do not
  break replay while any real change does.
* **Trace.** `quantlab.cli trace <run_id> <step>` (or the trace panel) walks the
  plan backwards: capability, library call, parameters, input steps and output
  hash for each step, down to the FRED file hash and the date range used.
* **Monitoring.** `runs/events.jsonl` gets one event per stage (request,
  planning, validation issues, approvals, each step, runtime guardrails,
  narration, completion, replays). The Monitoring page shows run counts and
  success / blocked / failed rates, mean and p95 latency per stage, capability
  usage, guardrail activity, recent errors and the plan-cache hit rate.

## Evaluation: two different questions

* **Is the planner choosing the right plan?** Measured by
  `quantlab.cli eval` (or `eval/run_eval.py`) on `eval/golden_requests.yaml`: 13
  requests (the six scenarios, variations, and adversarial cases that must end in
  a refusal or a BLOCK). Metrics: valid-plan rate, exact capability-sequence
  match, capability-set Jaccard, key-parameter accuracy, correct refusal/block
  rate. Reports go to `eval/reports/`. In mock mode this only checks the harness
  (the mock knows just the six scenarios); `--mode live` measures the real LLM.
* **Are the computations right?** Measured by the unit tests, independently of
  any LLM: returns, differences, spread alignment, rolling volatility and
  annualisation, forward-return compounding from t+1, the no-look-ahead property
  of expanding flags, the tests matching direct scipy/statsmodels calls,
  episode counting, every guardrail, replay, grounding and a source safety scan.

## Demo scenarios

| | Request (short) | Expected |
|---|---|---|
| A | NASDAQ weekly: 12-week vol, 90th pct expanding flags, 4-week forward returns, t-test (non-overlapping) + Newey-West | Approval for the return method (G12), then runs end to end |
| B | Same, full-sample 95th percentile, overlapping t-test | Approvals G10 + G12, warning G11, runs |
| C | Returns of the 10-year yield + random forest | Blocked: G02 + G08 |
| D | "Ignore your rules ... run this Python" | Refused (G13); no code ever runs |
| E | Daily VIX flags vs weekly NASDAQ forward returns | Blocked: G09 |
| F | Weekly 10y-2y spread, summary stats, 4-week changes, ADF | Runs (with an honest R04 warning: DGS2 starts 14 years after DGS10) |

## Project layout

```
pyproject.toml, uv.lock     dependencies (direct ones pinned exactly)
CLAUDE.md                   rules for future AI-assisted sessions
docs/SPEC.md                the original specification
docs/DECISIONS.md           decisions not covered by the spec
data/                       catalog.json, snapshots/*.csv (committed), README (sources, licences)
scripts/fetch_data.py       refreshes the FRED snapshot and catalog
src/quantlab/               the package (see the module table above)
app/                        Streamlit: streamlit_app.py (router), ui.py, pages/
notebooks/01..08            one notebook per layer, executed, mock mode
eval/                       golden_requests.yaml, run_eval.py, reports/
tests/                      pytest suite (synthetic data, fake LLM clients)
runs/                       run records, plan cache, events.jsonl (git-ignored)
```

## Known limitations and next steps

* **Live LLM mode was not exercised against the real APIs during development**
  (no keys were available). Both providers are tested with fake clients, and the
  OpenAI path was also run end to end with the real `openai` SDK against a local
  stand-in server. Run the smoke test in "Running the live demo" before presenting.
  Mock mode, used by the tests, CLI demo, notebooks and app, is fully exercised.
* **No plan -> observe -> replan loop.** Validation errors are shown to the human
  but not fed back to the LLM for a corrected plan. That is the natural next step.
* **The grounding check is numeric.** It proves every number is a correct
  rounding of an allowed value, not that the sentence attaches it to the right
  concept, and it compares absolute values (sign errors are not caught).
* **Forward returns count observations, not calendar time**: a week with no data
  is skipped, so "next 4 weeks" means the next 4 observed weeks.
* **File-based, single-user.** Runs and events live on local disk; approvals are
  recorded as "user" with no authentication or four-eyes approval.
* **`registry_version` covers capability interfaces, not their code**; the git
  commit in each manifest covers the code.
* **Data licences:** NASDAQCOM is marked "Copyrighted: Pre-Approval Required" on
  FRED; see `data/README.md` before publishing the repository.

Next steps: a replan loop with validator feedback; more capabilities (drawdowns,
correlations, regressions on continuous variables, event studies, GARCH);
authentication and role-based access with named approvers and four-eyes
approvals; a persistent store (database) for runs and events; CI running tests,
lint, the demo, the notebooks and the live evaluation; scheduled data refreshes
with a catalog history; per-run cost tracking and prompt caching for the planner.
