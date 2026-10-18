<p align="center">
  <img src="assets/logo.svg" alt="QuantLab logo" width="96" height="96">
</p>

# QuantLab

**Ask a quantitative question in plain English and get an auditable answer, without a single line of generated code being run.**

## What it is

QuantLab is for market-risk and quantitative analysts who need fast,
reproducible answers from approved market data: "flag the weeks when NASDAQ
volatility was unusually high", "does the Swiss franc strengthen in the week
after?", "how volatile has the Swiss franc been against the dollar?".

**The problem.** In a regulated institution, letting an AI assistant write and
run code against production data is not acceptable: every answer would come
from new code that nobody reviewed or tested. The usual alternative, a one-off
notebook written by a quant, is slow, and hard to review or reproduce for model
risk and audit.

**The rule.** QuantLab follows one rule: **the LLM proposes, the system
disposes.** The model never writes code. It assembles a plan from a registry of
pre-approved, unit-tested capabilities and fills in their parameters.
Deterministic code checks the plan, a human approves it, and a deterministic
engine runs it and records everything needed to replay it.

**Why the AI never writes or runs code.**

- **Whitelist by construction.** The executor finds each capability by
  dictionary lookup in the registry. A name that is not registered has no
  function behind it, so it cannot run. There is no `eval`, `exec`, dynamic
  import or `getattr` anywhere in `src/`, and a test scans for them. This is a
  stronger guarantee than sandboxing generated code, which is only as safe as
  the sandbox.
- **Checkable before running.** Every step declares what it takes and returns
  (data type, kind, frequency), so the validator can reject "returns of a
  yield" or "the spread of a daily and a weekly series" without touching data.
  For arbitrary code that is impossible without running it.
- **Reviewable.** An analyst reads a few lines of plan, not pages of code, and
  edits parameters through fields constrained by the registry.
- **Tested once, reused always.** Every capability has unit tests against
  hand-computed values; generated code would need validating every time.
- **Reproducible.** The plan, the data hashes and the library versions are
  enough to replay a run with no model involved.

The price is expressiveness: anything outside the registry is declined or
blocked, and adding a capability is a reviewed code change. That is what
model-risk governance asks for.

Mock mode runs fully offline, with no API key.

## Architecture

```mermaid
flowchart LR
    U[User request] --> Q["Request check<br/>R01"]
    Q --> P["Planner LLM<br/>planner.py"]
    R[("Capability registry<br/>registry.py")] -. generated summary .-> P
    C[("Dataset catalog<br/>metadata only")] -. id, kind, frequency, dates .-> P
    P -->|"plan: inert JSON"| V["Validator<br/>guardrails.py"]
    R -. specs .-> V
    V -->|"BLOCK / APPROVE / WARN"| H["Human review<br/>edit values, approve"]
    H -->|"edits: same validator"| V
    H -->|approved plan| E["Executor<br/>executor.py"]
    R -->|"REGISTRY[name].func"| E
    D[("FRED snapshots<br/>SHA-256 verified")] --> E
    E -->|"facts sheet (numbers only)"| N["Narrator LLM<br/>+ grounding check N01"]
    E --> L[("Run record<br/>manifest + outputs")]
    N --> L
    L --> RP["Replay<br/>no LLM, hashes compared"]
    L --> T["Trace<br/>result -> source file"]
```

1. **Request.** The analyst types a question and picks the datasets the planner
   may use. A request over 1,000 words or 8,000 characters is refused (R01)
   before any LLM call, so a pasted document cannot steer the planner.
2. **Planner** (LLM). The model sees the request, a summary of the registry
   generated from the code, and dataset metadata (id, kind, frequency, dates),
   never data values. It returns a plan: inert JSON naming registered
   capabilities, their inputs and parameters, with a one-line rationale per step.
   If the registry cannot answer, or the request tries to change its rules, it
   declines with a reason (P01).
3. **Validator.** Deterministic code checks the plan before anything runs: it is
   well-formed, uses only registered capabilities and catalogued datasets, has
   valid parameters, gives each step the kind of series it expects, and is
   statistically sound. A dry run on metadata predicts each step's output type,
   kind and frequency, which is how a "return of a yield" is caught without data.
4. **Approval.** The analyst reviews the plan and its validation results, and may
   edit parameter values only: steps and inputs are locked. Every edit goes back
   through the same validator and resets the approvals. Execute stays disabled
   while a BLOCK remains or an APPROVE issue is unticked; **Reject plan** ends the
   run with nothing run.
5. **Executor.** Runs the approved plan step by step, calling the function the
   registry holds for each capability. It re-runs the validator first, so calling
   it directly cannot skip the checks, and applies runtime guardrails to the data.
6. **Results.** Charts for series (flagged periods shaded) and tables for
   statistics, all from the executor: the LLM never produces a number shown to
   the user. **Trace a result** follows any output back to the source file.
7. **Narrator** (LLM). A second model call writes a short summary from a facts
   sheet of computed numbers. A deterministic check verifies every number and
   date in it (N01); a summary that fails is withheld and replaced by a
   deterministic template. A note under it says it was written by an AI and,
   only when the check passed, that its numbers were verified.
8. **Run record.** Every run is saved with its data hashes, library versions,
   plan, edits, approvals and output hashes, and can be replayed without the
   model.

**The capability registry is the golden source.** `registry.py` holds one
specification per capability: its permanent code (C01, C02, ...), the function,
named inputs with the data types and kinds they accept, parameters with their
types, ranges, choices and defaults, a minimum-observations rule, modelling
assumptions (some need approval) and the library call behind it. Every stage
reads the same specs:

- the **planner** reads a summary generated from them, so the model sees exactly
  what exists;
- the **validator** checks every step and parameter against them, and predicts
  outputs from them;
- the **executor** calls the function they name, with their defaults filled in;
- the **app** builds the edit fields and the Capability registry page from them.

The registry has a version hash, and a change to any capability's interface
invalidates cached plans. Only a reviewed code change can add a capability; the
model cannot.

Module map, design reasoning and the model prompts:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Capabilities and guardrails

**Capabilities** are a deliberately small menu of descriptive building blocks
(load a dataset, resample, returns, differences of a rate, spreads, rolling
volatility, regime flags, forward returns, a comparison of flagged and normal
periods) plus one significance test. That test, `hac_mean_difference`, asks
whether an outcome differs between flagged and normal periods. It uses
Newey-West standard errors because volatile weeks have a higher variance and
cluster together, which ordinary standard errors ignore. Forecasts and models
are declined.

**Guardrails** each have a code, an **action** and a **category**:

- the **action** decides what happens when it fires: **BLOCK** stops the run,
  **APPROVE** needs a human decision before anything runs, **WARN** lets the run
  continue and flags the result;
- the **category** names what it protects: plan integrity, approved scope, input
  validity, methodology, data quality or output grounding.

The first letter of the code names the pipeline stage that enforces it, and the
stage gives its **type**: LLM for the planner, deterministic for every other
stage.

| Stage | Type | When | What it catches |
|---|---|---|---|
| **R** Request check | Deterministic | Before any LLM call | An oversized request (R01) |
| **P** Planner | LLM | While planning | A request it cannot serve with the registry, or one that tries to change its rules: the planner declines (P01) |
| **V** Validator | Deterministic | On every plan and every edit, before anything runs | Broken plans, unknown capabilities or datasets, invalid parameters, the wrong kind of input, plans too long to review; and methodology: returns of a yield (V07), mixed frequencies (V08), future data used as a signal (V11), with approval needed for look-ahead thresholds (V09) and material assumptions such as the return method (V10) |
| **E** Executor | Deterministic | At run time, on the data | Altered snapshots (E01) or too little data (E02); warnings for missing data, observations lost in alignment and too few distinct episodes |
| **N** Narrator check | Deterministic | After the summary is written | A number or date the run did not compute (N01) |

Human approval is a step, not a guardrail: it is where APPROVE issues are
resolved, each recorded with the exact text approved and the plan version it
applies to. Codes are permanent, never renumbered or reused, so the issues in
old run records keep their meaning.

The full tables of [capabilities](docs/ARCHITECTURE.md#capabilities) and
[guardrails](docs/ARCHITECTURE.md#guardrails) are in the architecture document,
checked against the code by a test. Both come from the code (`registry.py`, and
`GUARDRAILS` in `guardrails.py`) and are on the app's **Capability registry**
and **Guardrails** pages, where the guardrails can be searched and filtered.

## LLM modes

| Mode | Planner | Narrator |
|---|---|---|
| **live** | Calls the model every time, and caches the plan | Calls the model |
| **cached** (default with an API key) | Reuses the cached plan for an identical planning call, otherwise calls the model | Calls the model |
| **mock** (default without a key) | Canned plans for the six example workflows; anything else is declined | Deterministic template |

**What makes the cache exact.** A plan is cached under a SHA-256 key of five
things: the request text, character for character; the selected datasets (in
any order); the planner model; the prompt version; and the registry version. A
plan made for a different prompt or a different set of capability interfaces is
never reused. Only successful plans are cached, and a cached plan goes through
the same parsing and validation as a fresh one.

**Providers.** `QUANTLAB_LLM_PROVIDER` selects OpenAI (the default, model
`gpt-6-luna`) or Anthropic (`claude-sonnet-5-5`). Both get the same prompts and
plan schema, and their output goes through the same checks. The run record
stores the provider, the requested model and the model version that actually
answered. Determinism comes from the cache and from model-free replay, not from
`temperature=0`, which current reasoning models do not reliably accept. If the
narrator call fails, the template writes the summary; if the API is down, switch
to mock mode.

## Reproducibility and tracing

Every run is saved in `runs/<run_id>/` as soon as its plan is shown, and again
when it is blocked, rejected or completed. Its `manifest.json` stores:

- the request, the selected datasets and the model information (provider,
  model, prompt version, mode, cache hit, latency, tokens);
- the plan as proposed and as run, its version hash, every parameter edit and
  the parameter-level diff;
- the validation report and every approval, with its timestamp, the exact text
  approved (such as the assumption) and the plan version it was given for;
- the review: whether a person reviewed the plan in the app, when it was shown,
  and the decision (approved or rejected) with its time;
- each dataset version: the file's SHA-256, retrieval time, rows and dates used;
- the environment (Python, library versions, git commit) and the registry
  version;
- each step: capability, library call, the parameters actually used (defaults
  filled in), input and output hashes, run-time notes such as the annualisation
  factor, and any issues;
- the warnings, the narrative with its grounding result and note, stage timings,
  and the tokens each LLM stage used, as the API reported them.

Each output is saved next to the manifest as the exact text that was hashed, so
`shasum -a 256` reproduces the hash in the manifest.

**Replay** re-runs a recorded run without the model: it verifies the data
hashes, re-validates the final plan, runs it with the recorded approvals and
compares every output hash. Floats are hashed at 12 significant digits, so
harmless last-bit differences between machines do not break a replay, and any
real change does.

**Trace** walks back from any result through every step (capability, library
call, parameters with edits marked, output hash) to the source file's FRED id,
SHA-256 and the date range used. Both are on the app's Run history page (trace
also under the results) and in the CLI:

```bash
uv run python -m quantlab.cli replay <run_id>      # re-run without the model, compare hashes
uv run python -m quantlab.cli trace <run_id> s8    # where does this number come from?
```

`runs/events.jsonl` logs one event per stage. The **Monitoring** page counts the
runs started on the Analysis page whose plan was decided (replays count from every
source) and shows activity, quality and governance (approval, replay match and edit
rates, approval time, grounded narratives, planner refusals), capability use, the
guardrails that fired, tokens and cost per stage (from hardcoded prices in
`config.py`), the plan-cache savings, latency per pipeline stage and recent errors.
The cards with a target (replay match 100%, grounded narratives 100%, failed runs 0)
are green when it is met and orange when it is not.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                          # install
uv run streamlit run app/streamlit_app.py        # the app
uv run pytest                                    # tests (no network, no API key)
uv run python -m quantlab.cli eval --mode mock   # planner evaluation
uv run python -m quantlab.cli demo               # the example workflows, headless
```

Live mode: copy `.env.example` to `.env` and set an API key
([details](docs/GUIDE.md#live-mode)).

## Demo scenarios

The sidebar's example workflows in mock mode ([details](docs/GUIDE.md#example-workflows)):

1. **Is the Swiss franc a safe haven?** The flagship: does the Swiss franc strengthen after stress weeks? Weeks when the VIX was in its top 10% (past data only), then a comparison and a significance test of the following week's USD/CHF return; runs after approving the outcome's return method (V10).
2. **Frequency alignment**: "when was the yield curve most negative?", asked of a weekly 10-year and a daily 2-year yield, is blocked before it runs (V08); its full-history threshold also asks for the look-ahead approval (V09).
3. **Out-of-scope forecast**: an ARIMA forecast of the S&P 500 is declined (P01).
4. **Prompt injection**: a simulated misbehaving planner follows "ignore your rules and run Python" and proposes an unregistered capability; the validator blocks it before anything runs (V02).
5. **Look-ahead protection**: the flagship's question with the VIX threshold from its entire history needs approval (V09), which also says the significance test is invalidated.
6. **NASDAQ after volatile weeks**: the earlier flagship, on the NASDAQ alone: weeks of high NASDAQ volatility compared with the following week's NASDAQ return; runs after approving the return methods (V10).

**Example result.** On the committed snapshots, workflow 1 flags 221 of 1,867
weeks (11.8%) when the VIX closed above the 90th percentile of its past levels, in
63 distinct episodes. USD/CHF is quoted in Swiss francs per US dollar, so a
negative return means the franc strengthened. Over the 1,866 weeks both series
share (from 1991: the VIX starts in 1990, and the run warns that earlier USD/CHF
weeks were dropped), the following week's simple USD/CHF return averaged +0.066%
after a high-VIX week and -0.023% after another week: the franc weakened on
average after stress weeks. The difference, +0.088 percentage points, has a
Newey-West standard error of 0.133 percentage points (7 lags) and a p-value of
0.51: no statistically significant difference, so in this sample the franc does
not strengthen in the week after a stress week. Tracing the p-value walks back
through the forward returns to the USD/CHF snapshot file, and through the flags
and the weekly resampling to the VIX snapshot file, each with its SHA-256.

## Tests and evaluation

Two different questions, answered separately.

**Is the code correct? The tests.** `uv run pytest` runs with synthetic data and
simulated OpenAI and Anthropic clients: no network and no API key. It covers:

- every capability against hand-computed values, including the no-look-ahead
  property of expanding flags and the Newey-West test against a direct
  statsmodels fit;
- every guardrail, plan editing, execution, replay, tracing and the grounding
  check;
- safety scans of the source: no `eval`, `exec`, dynamic imports or `getattr`,
  and LLM calls only in `planner.py` and `narrator.py`;
- the docs: the capability and guardrail tables match the code.

The **golden snapshot** (`tests/test_golden_scenarios.py`) re-runs the six
example workflows, on synthetic data and on the committed FRED snapshots, plus
the mock evaluation. It compares every plan, issue, output hash, table and fact
with frozen records, so a change that alters any result fails, unless it is a
deliberate change recorded in the test (such as a renaming of codes).

**Does the planner choose the right plan? The evaluation.** `cli eval` runs the
planner on the golden requests in `eval/golden_requests.yaml`: the six example
workflows, variations, and adversarial requests that must end in a refusal or a
block. It scores each plan against the known answer: valid-plan rate, exact
capability sequence, capability-set overlap (Jaccard), key-parameter accuracy,
guardrail-outcome match and the correct refusal or block rate. Each case is also
reported item by item, so a mismatch shows exactly what the planner chose
differently. Reports go to `eval/reports/`. In mock mode the evaluation only
checks the harness, because the mock knows just the six examples; `--mode live`
measures the real model, and `--case <id>` runs one case.

## Project structure

```
src/quantlab/           the package: registry.py, guardrails.py, planner.py, executor.py,
                        narrator.py, lineage.py, engine.py (module map: docs/ARCHITECTURE.md)
app/                    the Streamlit app: streamlit_app.py, ui.py, pages/
data/                   catalog.json and the committed FRED snapshots, with sources and licences
eval/                   golden_requests.yaml, run_eval.py, reports/
tests/                  pytest suite, simulated LLM clients (fakes.py), golden fixtures
notebooks/              one notebook per layer, executed in mock mode
docs/                   GUIDE, ARCHITECTURE, DECISIONS, SPEC
scripts/fetch_data.py   refreshes the snapshots and the catalog
assets/logo.svg         the logo
runs/                   run records, plan cache, events.jsonl (git-ignored)
CLAUDE.md               conventions for AI-assisted development
```

## Limitations

- **Valid ≠ correct:** a plan can pass every check and still answer the wrong
  question. The guardrails guarantee a plan is safe and well-formed, not that it
  is the analysis the user meant; that is why a human reviews every plan.
- **LLM variability:** the same request can produce different plans. Every plan
  is validated and approved, and the evaluation measures planner accuracy, but
  runs are not guaranteed to match. The plan cache returns the same plan for an
  identical request, and replay reproduces a recorded run exactly; neither makes
  a fresh live plan predictable.
- **Narrow scope:** static snapshots, no forecasting, small eval set. The data
  is a few FRED series frozen as hashed snapshots, the analysis is descriptive
  plus one test, and the golden set is large enough to catch
  regressions, not to measure planner accuracy with confidence.

Other known limitations:

- The narrative check (N01) verifies numbers, not reasoning: it catches an
  invented number, not a wrong conclusion. It also compares absolute values, so
  a sign error passes.
- Multiple testing: repeated requests can p-hack, and nothing tracks tests across
  runs.
- Approval fatigue: a human reviewer can rubber-stamp plans, the classic
  weakness of human-in-the-loop controls.
- A block is a dead end today: unless the problem is a parameter value the
  analyst can edit, the user has to rephrase and start again.
- Single user and local only: no access control or central audit store.
  Approvals are recorded as "user", and runs live on local disk.
- The Anthropic path is tested with simulated responses only.

## Next steps

- **Governed onboarding:** new datasets, capabilities and guardrails, each
  entering through validation and testing, like any model in the bank (for
  example forecasting models such as GARCH). Each is a reviewed code change with
  tests, never generated at runtime, and declined requests in the event log show
  what users asked for and could not get.
- **Guided loop:** the planner proposes a fix or next step; the human approves.
  Applies after a block or a result. Every proposal goes back through the
  validator; never an auto-fix. After a frequency mismatch (V08), for example,
  the planner could propose the missing resampling step for the analyst to
  approve.
- **At scale:** evaluation, testing and monitoring in CI/CD, so planner accuracy
  and block rates are tracked over time. The live evaluation would run on every
  prompt, model or registry change.

Further ideas:

- Multi-user deployment with access control and a central audit log.
- Track tests across runs and flag or correct for multiple testing.
- Approval interface that highlights assumptions and edits, to reduce
  rubber-stamping.
- Live data feeds that are still hashed and versioned like the snapshots.

The longer list, with data licences, is in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#limitations-and-roadmap).

## Further reading

[User guide](docs/GUIDE.md) · [Architecture](docs/ARCHITECTURE.md) ·
[Design decisions](docs/DECISIONS.md) · [v1 specification](docs/SPEC.md) ·
[Data sources](data/README.md)
