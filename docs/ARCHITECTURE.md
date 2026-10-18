# QuantLab architecture

How QuantLab is built and why: the safety model, the pipeline and its modules,
the capability and guardrail tables, and the model calls. The
[README](../README.md) gives the overview, [GUIDE.md](GUIDE.md) explains how to
use it, and [DECISIONS.md](DECISIONS.md) records the rationale of every decision.

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
  weekly data, a signal built from future returns, a significance test on a price
  level) or require explicit approval (look-ahead thresholds, the return method),
  each with a written reason;
* requests outside the registry (a forecast, a model, running code) are declined,
  never improvised;
* every number shown comes from the engine, and the narrative is checked against it;
* every result can be traced back to the exact source file and replayed.

QuantLab also runs fully offline in **mock mode**, which needs no API key.

## Pipeline and modules

```mermaid
flowchart LR
    U[User request] --> P["Planner LLM<br/>planner.py"]
    R[("Capability registry<br/>registry.py")] -. generated summary .-> P
    C[("Dataset catalog<br/>metadata only")] -. id, kind, frequency, dates .-> P
    P -->|"plan: inert JSON"| V["Validator<br/>static guardrails<br/>guardrails.py"]
    V -->|"BLOCK / APPROVE / WARN"| H["Human review<br/>edit params, approve"]
    H -->|"edits: same validator"| V
    H -->|approved plan| E["Executor<br/>runtime guardrails<br/>executor.py"]
    R -->|"REGISTRY[name].func"| E
    D[("FRED snapshot<br/>SHA-256 verified")] --> E
    E -->|"facts sheet (numbers only)"| N["Narrator LLM<br/>+ grounding check (N01)"]
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
| `capabilities.py` | The quantitative functions: the only code that calls pandas/numpy/statsmodels for analysis |
| `registry.py` | Capability specs, the `REGISTRY` whitelist, generated LLM summary and registry version |
| `plan.py` | Pydantic `Plan` / `Step` (extra fields forbidden), plan version hash, the LLM tool schema |
| `guardrails.py` | The guardrail table (action, enforcer, category), validator guardrails (G codes), L01 refusal block, edit lock, metadata dry run, `ValidationReport` |
| `mock_scenarios.py` | Example workflows 1-5 with their canned plans (mock mode, CLI, app, tests) |
| `executor.py` | Registry dispatch, approval gate, runtime guardrails (R codes), step records |
| `planner.py` | OpenAI and Anthropic planners, mock planner, plan cache |
| `narrator.py` | Facts sheet, LLM narrator, template fallback, numeric grounding check (N01) |
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
* **Tested once, reused always.** Every capability has unit tests against
  hand-computed values; generated code would need validating every time.
* **Reproducible.** Plan, data hashes and library versions are enough to replay a
  run with no model involved.

The trade-off is expressiveness: anything outside the registry is declined (L01)
or blocked (G02). Adding a capability is a deliberate, reviewed code change,
which is what model-risk governance requires.

## Capabilities

The approved menu, deliberately small: descriptive building blocks and one
significance test; the **Capability registry** page always shows the current list. Each capability has a permanent
code (C01, C02, ...) used in the app, run records and traces; plans and the
planner use its name. The codes were renumbered once, when the registry was
simplified (`docs/DECISIONS.md` lists the old and new codes), and are permanent
from that version on.

| Code | Capability | What it does |
|---|---|---|
| C01 | `load_series` | Load one dataset from the catalog (hash-verified, missing days dropped) |
| C02 | `resample` | Daily to weekly (Friday) or monthly (month end), or keep daily (`D`, unchanged) |
| C03 | `compute_returns` | Simple or log returns of a price or index |
| C04 | `difference` | Changes of a rate or yield, in its own units |
| C05 | `spread` | a - b on the dates both series share |
| C06 | `rolling_volatility` | Trailing rolling standard deviation, annualised |
| C07 | `flag_regime` | Flag periods above a quantile threshold (e.g. high volatility) |
| C08 | `forward_return` | The return from the close of t to the close of t+1 (the next period), stamped at t (an outcome only) |
| C09 | `compare_groups` | Mean and standard deviation of a series in flagged vs normal periods, and the number of flagged episodes (descriptive, no test) |
| C10 | `hac_mean_difference` | Significance test of the difference in mean outcome between flagged and normal periods (Newey-West standard errors) |

**What follows a flag, and whether it is significant.** `forward_return` gives,
at each date t, the return from the close of t to the close of t+1, one period
of the series' frequency (the following week on weekly data; the last date has
no future price yet and is dropped, never filled). It is future information at
t, so it may only be the outcome of `compare_groups` or `hac_mean_difference`:
feeding it to anything else, such as a volatility or a flag, is blocked (G11).
`hac_mean_difference` regresses the outcome on a constant and the 0/1 flag,
which makes the flag's coefficient exactly the difference in means, and uses
Newey-West (HAC) standard errors, because ordinary ones assume independent weeks
with equal variance: flagged weeks have a much higher variance than normal weeks
(heteroskedasticity), and volatility clusters, so neighbouring weeks are not
independent (autocorrelation). Its lag count is always the Newey-West rule
floor(4·(T/100)^(2/9)). The outcome must be returns or forward returns (a
level, which is non-stationary: G04). The test assumes a
stationary outcome, a large sample (its p-value is asymptotic) and flags known
before the outcome; it reports an association, never a cause or a forecast.
Every comparison also counts the distinct flagged episodes, the real amount of
independent evidence (fewer than 5: R05).

There are no other statistical tests, no regressions beyond this one and no
forecasts: such requests are declined or blocked (example 3). Each spec declares named inputs (allowed data
types and kinds), parameters (type, range, choices, defaults), output type and
kind, a minimum-observations rule, assumptions (some need approval) and the
underlying library call.

**A deliberately small library surface:** pandas and numpy (time series),
statsmodels (the regression with Newey-West standard errors behind the
significance test), pydantic (strict plan parsing), streamlit and plotly (UI),
the official `openai` and `anthropic` SDKs, python-dotenv and pyyaml.
Deliberately excluded: scipy as a direct dependency (statsmodels installs it;
QuantLab never calls it), scikit-learn (no machine-learning models),
`arch` (no GARCH), and LangChain or any agent framework (two structured model
calls do not need an orchestration layer, and every extra abstraction is more to
audit). Fewer libraries means fewer versions to pin, record and explain.

## Guardrails

Each guardrail is a small named function whose docstring explains what it checks
and what goes wrong without it. Every guardrail has:

* an **action**, what happens when it fires: **BLOCK** (the run stops),
  **APPROVE** (a human must decide before anything runs) or **WARN** (the run
  continues and the result is flagged);
* an **enforcer**, given by the first letter of its code: **L** the planner
  (LLM), which may refuse; **G** the validator, deterministic, on the plan alone
  (the planner's plan and every user edit); **R** the executor, at run time on
  the data; **N** the narrator check, deterministic, on the written summary;
* a **category**, what it protects: plan integrity, approved scope, input
  validity, methodology, data quality or output grounding.

The table matches `guardrails.GUARDRAILS` row for row (`tests/test_docs.py`
checks it); the same table feeds the **Guardrails** page, its counts and every
issue shown in the app. The codes were
renumbered twice while the design settled (`docs/DECISIONS.md` lists the old and
new codes); from this version on they are permanent: never renumbered or given
another meaning, so the issue ids in run records keep their meaning and those
runs can still be replayed.

| Code | Name | Action | Enforced by | Category | Description |
|---|---|---|---|---|---|
| L01 | `PLANNER_REFUSED` | BLOCK | Planner (LLM) | Approved scope | The planner's system prompt explicitly allows it to refuse a request it cannot fulfil with the registry, or one that tries to change its rules, giving a reason instead of a plan; the refusal is blocked and nothing runs. This prevents the LLM from forcing an out-of-scope or malicious request into a misleading plan. |
| G01 | `INVALID_PLAN` | BLOCK | Validator (deterministic) | Plan integrity | Checks the plan is well-formed: it matches the plan schema exactly (no extra fields), has steps, has unique step ids, and every input points to an earlier step. Also raised when the planner returns no usable plan (API error, cut-off answer, provider refusal, no plan submitted). This prevents running a broken, ambiguous or circular plan. |
| G02 | `UNKNOWN_CAPABILITY` | BLOCK | Validator (deterministic) | Approved scope | Checks every step calls a capability in the approved registry. This prevents the system ever running code that nobody wrote, reviewed or tested, whatever the LLM asks for. |
| G03 | `INVALID_PARAM` | BLOCK | Validator (deterministic) | Input validity | Checks each parameter is known, given when required, of the right type and within its allowed range or choices, for example a quantile between 0.5 and 0.99. Edits that try to change anything other than a parameter value are rejected under this code too. This prevents nonsensical settings from producing plausible-looking numbers. |
| G04 | `TYPE_MISMATCH` | BLOCK | Validator (deterministic) | Input validity | Checks each step receives the kind of series it expects, for example volatility computed on returns, not on prices, and a significance test on returns, not on a price level. This prevents statistically meaningless calculations. |
| G05 | `BUDGET_EXCEEDED` | BLOCK | Validator (deterministic) | Plan integrity | Checks the request length, before the planner is called (so an oversized request costs no LLM call) and again with every plan, and the number of steps in the plan. This prevents oversized requests and runaway plans too long for a human to review. |
| G06 | `UNKNOWN_DATASET` | BLOCK | Validator (deterministic) | Approved scope | Checks every dataset the plan names or loads is in the approved, hash-versioned catalog. This prevents the use of unverified or invented data. |
| G07 | `RETURNS_ON_RATES` | BLOCK | Validator (deterministic) | Methodology | Checks that returns aren't computed on interest rates, which call for differences instead. This prevents economically wrong results, such as percentage changes of a yield near zero. |
| G08 | `FREQUENCY_MISMATCH` | BLOCK | Validator (deterministic) | Methodology | Checks that series combined in one step (a spread, a comparison, a test) share the same frequency. This prevents silently comparing weekly values with daily ones. |
| G09 | `LOOK_AHEAD` | APPROVE | Validator (deterministic) | Methodology | Detects thresholds built from the full sample, including future data, and asks for approval. This prevents a result that looks predictive only because it used information unavailable at the time. |
| G10 | `MATERIAL_ASSUMPTION` | APPROVE | Validator (deterministic) | Methodology | Asks for human approval when a choice changes the answer, today the return method (log versus simple returns). This prevents important assumptions being made silently. |
| G11 | `FORWARD_DATA_AS_SIGNAL` | BLOCK | Validator (deterministic) | Methodology | Checks forward returns are used only as outcomes, never to build a flag or signal. This prevents the future leaking into the explanation of the future. |
| R01 | `DATASET_HASH_MISMATCH` | BLOCK | Executor (runtime) | Data quality | Checks each loaded snapshot matches the hash recorded in the catalog. This prevents the analysis running on altered or different data. |
| R02 | `INSUFFICIENT_DATA` | BLOCK | Executor (runtime) | Data quality | Checks each step has enough observations for its calculation, for example three times the rolling window. This prevents statistics computed on too little data. |
| R03 | `MISSING_DATA` | WARN | Executor (runtime) | Data quality | Warns when more than 5% of a loaded series' values are missing (they are dropped). This prevents gaps silently distorting the results. |
| R04 | `ALIGNMENT_LOSS` | WARN | Executor (runtime) | Data quality | Warns when aligning two series on their common dates drops more than 10% of either. This prevents hidden sample shrinkage. |
| R05 | `LOW_EFFECTIVE_SAMPLE` | WARN | Executor (runtime) | Methodology | Warns when the flagged periods of a comparison or test come from fewer than 5 distinct episodes. This prevents overconfidence in a result that looks large but rests on few real events. |
| N01 | `UNGROUNDED_NARRATIVE` | WARN | Narrator check (deterministic) | Output grounding | Checks deterministically that every number and date in the written summary is a correct rounding of a value the run computed (the facts sheet) or used (plan parameters, the request, warnings, dataset titles). An LLM summary that fails is withheld and the deterministic summary shown instead, with the failing numbers listed. This prevents the narrative from inventing or misquoting numbers, so every figure can be traced back to data, functions and parameters. |

The executor re-runs the validator itself and refuses to start while any BLOCK
exists or any APPROVE issue is unapproved, so calling it directly cannot skip
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
`mock` (canned plans for the five example workflows; anything else is politely
declined). Mock mode is also the fallback whenever the network or the API is
unavailable.

**Narrator** (`narrator.py`). Receives the request, a plain plan summary,
warnings, approvals and a *facts sheet* of rounded computed numbers, and must use
only those numbers. `check_grounding` then extracts every number in the narrative
and checks that it is a correct rounding of a fact (or its percentage form), a
plan parameter, a number from the request or a warning, or an annualisation
factor. Dates are facts too: a date, as `2021-11-19` or in words such as
"19 November 2021", must be a date of the facts sheet (a month must contain one,
and a year on its own must lie within the data), so a narrative that names the
wrong peak date fails the check. For the significance test the narrator must
state the difference and its p-value, write "no statistically significant
difference" when the p-value is 0.05 or more (never "no difference"), always
give the number of distinct episodes and never say that the flag causes or
predicts returns; only then are the 5% significance and 95% confidence levels
accepted numbers. That check is guardrail N01 UNGROUNDED_NARRATIVE: when it
passes the app shows "all numbers verified"; when a model-written narrative fails
it, the narrative is withheld (kept in the run record), the deterministic template
narrative is shown instead, and an N01 warning lists the numbers that failed. The
numerical results are not affected, and the deterministic facts table is always
shown. Directly under the narrative, the app and the CLI show a short note, also
kept in the run record (`narrator.AI_WRITTEN_NOTE` and `NUMBERS_CHECKED_NOTE`): an
LLM narrative "was written by an AI and may contain errors in wording or
reasoning", and, only when N01 passed, "every number in it has been checked
automatically against the computed results (N01)".

## Limitations and roadmap

* **Live mode coverage.** The OpenAI path is in use with `gpt-6-luna` for both
  the planner and the narrator. The Anthropic path is covered by automated tests
  with simulated responses only, and the full live planner evaluation (`eval
  --mode live`) has not yet been run; the flagship case (with both the longer
  and the current, shorter request) was run live three times and matched the
  canned plan exactly each time. Mock mode, used by the tests, the CLI, the
  notebooks and the app, is fully covered.
* **No plan -> observe -> replan loop.** Validation errors are shown to the
  analyst but not yet fed back to the model for a corrected plan.
* **The grounding check is numeric.** It proves every number is a correct
  rounding of an allowed value and every date is a date of the results, not that
  the sentence attaches it to the right concept (a true date described as the
  wrong extreme would pass), and it compares absolute values (sign errors are
  not caught).
* **One significance test, no forecasts.** QuantLab describes what happened
  (averages, volatility, high-volatility periods, spreads) and can test one kind
  of question, whether an outcome differs between flagged and normal periods,
  with an asymptotic p-value. Other tests, models and forecasts are declined or
  blocked.
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
  correlations, further statistical tests and regressions, GARCH), each added as a code
  change with tests, never generated at runtime.
* **New guardrails.** Further validation rules as new risks are identified, each with a
  defined action (BLOCK, APPROVE or WARN), enforcer, category and test coverage.
* **Replan loop.** Validator feedback sent back to the planner for a corrected plan.
* **Access control.** Authentication and role-based access, with named approvers and
  four-eyes approvals.
* **Persistent store** for runs and events.
* **Continuous integration** running the tests, lint, the example workflows, the
  notebooks and the live evaluation.
* **Cost control.** Per-run cost tracking and prompt caching for the planner.
