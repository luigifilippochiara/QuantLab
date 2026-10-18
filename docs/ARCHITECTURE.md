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
   and dataset metadata, never data values. An oversized request is refused
   before the model is called (R01).
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
| `config.py` | Settings from `.env`: API key, models, LLM mode, paths, budgets; LLM prices for the cost shown on the Monitoring page |
| `types.py` | `DataType`, `SeriesKind`, `Frequency`; `Artifact` (value + explicit metadata); canonical hashing; `TokenUsage` (tokens of an LLM stage) |
| `data.py` | Catalog, FRED CSV parsing, SHA-256 verification, LLM-safe dataset metadata |
| `capabilities.py` | The quantitative functions: the only code that calls pandas/numpy/statsmodels for analysis |
| `registry.py` | Capability specs, the `REGISTRY` whitelist, generated LLM summary and registry version |
| `plan.py` | Pydantic `Plan` / `Step` (extra fields forbidden), plan version hash, the LLM tool schema |
| `guardrails.py` | The guardrail table (action, enforcer, category), request check (R01), validator guardrails (V codes), P01 refusal block, edit lock, metadata dry run, `ValidationReport` |
| `mock_scenarios.py` | Example workflows 1-6 with their canned plans (mock mode, CLI, app, tests) |
| `executor.py` | Registry dispatch, approval gate, runtime guardrails (E codes), step records |
| `planner.py` | OpenAI and Anthropic planners, mock planner, plan cache |
| `narrator.py` | Facts sheet, LLM narrator, template fallback, numeric grounding check (N01) |
| `lineage.py` | Run record (manifest), save/load, replay, trace |
| `monitoring.py` | Append-only JSONL event log; the Monitoring page's metrics, from run records and events, and its attention thresholds |
| `engine.py` | `Session`: plan, validate, edit, approve, reject or execute, narrate, save; used by app, CLI, notebooks |
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

The trade-off is expressiveness: anything outside the registry is declined (P01)
or blocked (V02). Adding a capability is a deliberate, reviewed code change,
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
feeding it to anything else, such as a volatility or a flag, is blocked (V11).
`hac_mean_difference` regresses the outcome on a constant and the 0/1 flag,
which makes the flag's coefficient exactly the difference in means, and uses
Newey-West (HAC) standard errors, because ordinary ones assume independent weeks
with equal variance: flagged weeks have a much higher variance than normal weeks
(heteroskedasticity), and volatility clusters, so neighbouring weeks are not
independent (autocorrelation). Its lag count is always the Newey-West rule
floor(4·(T/100)^(2/9)). The outcome must be returns or forward returns (a
level, which is non-stationary: V04). The test assumes a
stationary outcome, a large sample (its p-value is asymptotic) and flags known
before the outcome; it reports an association, never a cause or a forecast.
Every comparison also counts the distinct flagged episodes, the real amount of
independent evidence (fewer than 5: E05).

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
* an **enforcer**, the pipeline stage named by the first letter of its code, in
  pipeline order: **R** the request check, deterministic, before the planner is
  called; **P** the planner (LLM), which may refuse; **V** the validator,
  deterministic, on the plan alone (the planner's plan and every user edit);
  **E** the executor, at run time on the data; **N** the narrator check,
  deterministic, on the written summary. Human approval is not a guardrail: it is
  the step where APPROVE issues are resolved;
* a **type**, derived from the stage: **LLM** for the planner's guardrail,
  **Deterministic** for every other stage;
* a **category**, what it protects: plan integrity, approved scope, input
  validity, methodology, data quality or output grounding
  (`guardrails.CATEGORY_DEFINITIONS`, also in the Guardrails page's Category filter):
  - **Plan integrity**: The plan is well-formed and of a size a person can review.
  - **Approved scope**: Only approved capabilities and catalogued datasets, for requests the registry can serve.
  - **Input validity**: Every input is acceptable: the request's size, each parameter, each step's input kind, and data files that match their recorded hash.
  - **Methodology**: The analysis is statistically and economically sound.
  - **Data quality**: There is enough data, and gaps or alignment do not silently distort it.
  - **Output grounding**: The written summary quotes only values the run computed or used.

The table matches `guardrails.GUARDRAILS` row for row (`tests/test_docs.py`
checks it); the same table feeds the **Guardrails** page, its counts and every
issue shown in the app. The codes were
renumbered twice while the design settled (`docs/DECISIONS.md` lists the old and
new codes); from this version on they are permanent: never renumbered or given
another meaning, so the issue ids in run records keep their meaning and those
runs can still be replayed.

| Code | Name | Action | Enforced by | Type | Category | Description |
|---|---|---|---|---|---|---|
| R01 | `REQUEST_TOO_LONG` | BLOCK | Request check | Deterministic | Input validity | Blocks a request over 1,000 words, or 8,000 characters (so a text without spaces cannot slip through), before it reaches the LLM. Prevents pasted documents or stuffed instructions from driving the planner. |
| P01 | `PLANNER_REFUSED` | BLOCK | Planner | LLM | Approved scope | The planner may decline a request it cannot serve with the approved capabilities, or one that tries to override its rules, giving a reason instead of a plan (in mock mode, any request without a canned plan); nothing runs. Prevents an out-of-scope or malicious request from being forced into a misleading plan. |
| V01 | `INVALID_PLAN` | BLOCK | Validator | Deterministic | Plan integrity | The plan must match the plan schema exactly (no missing or extra fields), have at least one step, use unique step ids, and give each step exactly the inputs its capability defines, each taken from an earlier step. Also raised when the planner returns no usable plan: an API error or timeout, an answer cut off or refused by the provider, or no readable plan submitted. Prevents a malformed, ambiguous or out-of-order plan from running. |
| V02 | `UNKNOWN_CAPABILITY` | BLOCK | Validator | Deterministic | Approved scope | Every step must call a capability in the approved registry, so no unreviewed or untested code is ever run, whatever the LLM asks for. |
| V03 | `INVALID_PARAM` | BLOCK | Validator | Deterministic | Input validity | Each parameter must be one its capability defines, given if required, of the right type and within its allowed range or choices (for example a quantile between 0.5 and 0.99). User edits that change anything other than a parameter value, such as a step, an input or the capability, are rejected too. Prevents nonsensical settings from producing plausible-looking numbers. |
| V04 | `TYPE_MISMATCH` | BLOCK | Validator | Deterministic | Input validity | Each input must be the data type and kind of series its capability accepts: for example returns, not prices, for a volatility, and returns, not a price or rate level, for the significance test. Prevents statistically meaningless calculations. |
| V05 | `BUDGET_EXCEEDED` | BLOCK | Validator | Deterministic | Plan integrity | The plan may have at most 15 steps by default (QUANTLAB_MAX_STEPS). Prevents runaway plans too long for a human to review. |
| V06 | `UNKNOWN_DATASET` | BLOCK | Validator | Deterministic | Approved scope | Every dataset the plan names or loads must be in the approved catalog. Prevents analysis of invented or unapproved data. |
| V07 | `RETURNS_ON_RATES` | BLOCK | Validator | Deterministic | Methodology | Returns cannot be computed on an interest rate or yield: rates are differenced instead (changes in percentage points). Prevents meaningless results, such as a move from 1% to 2% reported as +100%. |
| V08 | `FREQUENCY_MISMATCH` | BLOCK | Validator | Deterministic | Methodology | Series combined in one step (a spread, a comparison or the significance test) must have the same frequency. Prevents silently matching weekly values with daily ones on the few dates they share. |
| V09 | `LOOK_AHEAD` | APPROVE | Validator | Deterministic | Methodology | A flag threshold computed from the full sample, which includes data from after each flagged date, needs explicit approval. Prevents results that look better than anything achievable in real time from passing unnoticed. |
| V10 | `MATERIAL_ASSUMPTION` | APPROVE | Validator | Deterministic | Methodology | Choices that change the answer need explicit approval: today, the return method (log or simple) of compute_returns and forward_return. Prevents the LLM from making material assumptions silently. |
| V11 | `FORWARD_DATA_AS_SIGNAL` | BLOCK | Validator | Deterministic | Methodology | Forward returns may only be the outcome of a comparison or of the significance test, never the input to a volatility, a flag or any other step. Prevents future information from leaking into a signal. |
| E01 | `DATASET_HASH_MISMATCH` | BLOCK | Executor | Deterministic | Input validity | Each snapshot file must match the hash recorded in the catalog before it is loaded. Prevents analysis of altered or different data. |
| E02 | `INSUFFICIENT_DATA` | BLOCK | Executor | Deterministic | Data quality | Each step needs its capability's minimum number of observations, for example three times the window for a rolling volatility and 30 shared dates for the significance test. Prevents statistics computed on too little data. |
| E03 | `MISSING_DATA` | WARN | Executor | Deterministic | Data quality | Warns when more than 5% of a loaded series' values are missing (they are dropped). Prevents gaps from silently distorting the results. |
| E04 | `ALIGNMENT_LOSS` | WARN | Executor | Deterministic | Data quality | Warns when aligning two series on their common dates drops more than 10% of either. Prevents a silently shrunken sample. |
| E05 | `LOW_EFFECTIVE_SAMPLE` | WARN | Executor | Deterministic | Methodology | Warns when the flagged periods of a comparison or of the significance test form fewer than 5 distinct episodes (runs of consecutive flags). Prevents overconfidence in a result that rests on a handful of real events. |
| N01 | `UNGROUNDED_NARRATIVE` | WARN | Narrator check | Deterministic | Output grounding | Every number in the written summary must be a correct rounding of a value the run computed or used: results, parameters, the request, warnings, dataset titles, annualisation factors and, when a test ran, the 5% and 95% levels. Dates must be dates of the results (a year on its own, within their range). A failing LLM summary is withheld and the deterministic one, checked the same way, is shown instead, with the failing numbers listed. Prevents invented or misquoted figures. |

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
and is then parsed strictly (V01) and validated. Two API constraints shaped this
(details in `docs/DECISIONS.md`): current reasoning models do not reliably accept
a *forced* tool choice or a non-default temperature, so the tool call is
requested by instruction (`tool_choice="auto"`), reasoning depth is set with an
effort level, and determinism comes from the plan cache and model-free replay
rather than `temperature=0`.

**Modes:** `live` (call the API), `cached` (reuse the cached plan for an
identical request, datasets, model, prompt version and registry version), and
`mock` (canned plans for the six example workflows; anything else is politely
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
