# QuantLab user guide

How to install, run and use QuantLab day to day. The [README](../README.md) gives
the overview, [ARCHITECTURE.md](ARCHITECTURE.md) explains the design and holds the
capability and guardrail tables, and [DECISIONS.md](DECISIONS.md) records why each
choice was made.

## Intended users and use cases

QuantLab is built for **market-risk and quantitative analysts** who need fast,
reproducible answers from approved market data. Typical questions:

* "Flag the weeks when NASDAQ volatility was unusually high versus its own history."
* "Was the average weekly NASDAQ return different in high-VIX weeks?"
* "Where is the 10-year minus 2-year Treasury spread today, and how volatile has it been?"
* "How volatile has the Swiss franc been against the dollar over rolling 26-week windows?"

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
uv run python -m quantlab.cli eval --mode live --case s1_volatility_flag_success   # one case, item by item

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
   and its main parameters (e.g. "Step 4 · C06 · Rolling volatility · window: 12
   weeks, annualize: yes"); open it for its inputs, predicted output and editable
   fields. Steps with a blocking issue or a warning open by default with a badge,
   and edited steps are marked. Approvals are not shown on the steps: each one is
   listed in the validation results, naming its step (e.g. "Step 3 · C03 · Compute
   returns").
3. Read the validation results. Blocking issues must be resolved, approvals must
   be ticked, and warnings are shown for context.
4. Click **Execute**. Results appear per step: charts for series (with flagged
   periods shaded), tables for statistics, and a plain-English narrative marked
   "all numbers verified" when every number checks out. A short note under the
   narrative says who wrote it (an AI narrative may contain errors in wording or
   reasoning) and, only when the check passed, that every number was checked (N01).
5. Use **Trace a result** under the results to follow any output back through
   every step to the source file, with its SHA-256, FRED id and retrieval date.

The other pages: **Run history** lists past runs, shows each manifest and user
edits, and replays a run without the model; **Monitoring** shows run outcomes,
latency per stage, capability usage and guardrail activity; **Capability
registry** lists every approved capability with its code, inputs, parameters and
assumptions; **Guardrails** lists every guardrail with its action, who enforces
it, its category and what it prevents.

### Editing a plan before it runs

After a plan is generated, each step's parameters appear as fields generated
from the registry (select boxes for choices, bounded number inputs, checkboxes),
prefilled with the planner's values. Only values can change: steps, capabilities
and input references are locked, and the engine rejects any other edit with a
BLOCK. **Apply changes & re-validate** runs the same validator again, so issues
appear and clear as the plan changes: in example 1, switching
`flag_regime.method` to `full_sample` raises G09, which then also says the
significance test is invalidated, and changing `load_series.dataset_id` from
NASDAQCOM to DGS10 is a legal edit that the validator then blocks (G07: returns
of a yield are meaningless). Windows are counted in periods of the
series' frequency, as their fields and the step summaries say: switching
`resample.freq` from W to D keeps `window` at 12, which then means 12 trading
days instead of 12 weeks. Any edit resets the
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

1. **High-volatility weeks and what follows** (`volatility_flag_success`). The
   request states the choices that have no default; the registry defaults supply
   the rest (annualised volatility, an expanding 90th-percentile threshold after
   52 weeks), and forward returns always cover the next week with the test's lags
   set by the Newey-West rule, so the live planner can reach the same plan as the
   canned one:

   > Using NASDAQ, calculate weekly log returns, 12-week volatility, and flag unusually volatile weeks. Compare next-week simple returns after volatile weeks with those after normal weeks, then test whether the difference in means is statistically significant.

   Weekly Friday closes; weekly log returns for the volatility (they add up over
   time, as square-root-of-time annualisation assumes); 12-week annualised
   rolling volatility; flags above the 90th percentile of past volatility only
   (expanding, after 52 weeks); the following week's simple return stamped on
   each week (`forward_return`: a mean of log returns would be lower by about
   half the variance and bias the comparison against volatile weeks); a
   descriptive comparison first (`compare_groups`), then the significance test
   (`hac_mean_difference`). The analyst confirms the two return-method choices
   (G10); there is no warning and no block. The result shows the volatility with
   the high-volatility weeks shaded, both groups' means and standard deviations,
   the number of distinct episodes, and the test's difference, p-value and 95%
   confidence interval.
2. **Look-ahead protection** (`vix_full_sample_lookahead`): "Flag the days when
   the VIX was in the top 5% of its entire history." The plan flags the days above
   the 95th percentile of the whole VIX history, a full-sample threshold that uses
   data nobody had on the flagged day. QuantLab runs it only with explicit approval
   (G09), explains that such a threshold judges past days using information from
   the future, and recommends the expanding method. It is an approval, not a
   block: a full-sample threshold is fine for labelling past stress days but
   invalid as a signal or in a backtest, and only the analyst knows which is meant.
3. **Out-of-scope forecast request** (`arima_forecast_block`): "Fit an ARIMA
   model to forecast next week's USD/CHF exchange rate." The registry has no
   capability that fits a model or makes a forecast, so the planner declines
   (L01) and lists the registered capabilities instead. If a planner ever
   proposed such a step anyway, the validator would block it independently (G02).
4. **Prompt-injection protection** (`prompt_injection_block`): a request that
   tells the assistant to ignore its rules and run Python is treated as data,
   not as instructions, and declined (L01). There is no code path that could run
   it, and automated tests confirm that nothing from the request reaches a shell
   or a subprocess.
5. **Frequency-alignment protection** (`spread_frequency_mismatch_block`):
   "Compute the spread between the weekly 10-year Treasury yield and the daily
   2-year Treasury yield." Planned exactly as asked, the spread of a weekly and a
   daily series would silently keep only the Fridays and mix two frequencies. The
   validator, not the planner, blocks it (G08) before anything runs and states the
   fix: resample the 2-year yield to weekly first.

**Example output.** For example workflow 1 on the 1971-2026 NASDAQ snapshot,
QuantLab reports that 446 of 2,842 weeks (15.7%) had a 12-week annualised
volatility above the 90th percentile of their own past, in 40 distinct episodes.
The following week's simple return averaged 0.209% after a flagged week and
0.230% after a normal week: a difference of -0.021 percentage points, with a
Newey-West standard error of 0.223 (8 lags) and a p-value of 0.93, so there is no
statistically significant difference. The share of flagged weeks exceeds 10%
because each week is compared only with the history available at the time, so
the more volatile decades after the 1970s cross the earlier thresholds more often.

**Audit trail.** For any executed result, such as example 1's p-value, **Trace a
result** shows the full chain `hac_mean_difference` -> `forward_return` and
`flag_regime` -> `rolling_volatility` -> `compute_returns` -> `resample` ->
`load_series` -> the NASDAQ snapshot file, with each step's library call,
parameters (user edits marked) and output hash. `cli demo` prints the same chain.

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
uv run python -m quantlab.cli run "Flag the days when the VIX is above its 95th percentile, using only past data." --datasets VIXCLS --mode live
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
canned plan (for example refuse example 5 outright with L01, rather than
proposing the misaligned spread that the validator blocks with G08); the command
only checks that blocked workflows stay blocked.

**4. Start the app**, open http://localhost:8501 and pick **cached** (the default
when a key is set) or **live** in the sidebar. If the API is unavailable, switch
to **mock**: everything keeps working offline with the canned plans.

## Reproducibility, lineage and monitoring

* **Run record.** `runs/<run_id>/manifest.json`: request, model information
  (model, prompt version, mode, cache hit, latency, tokens), the plan as proposed
  and as run with its version hash, parameter edits and the parameter-level diff,
  validation report, approvals, dataset versions (file SHA-256, retrieval time,
  rows and dates used), environment (Python, library versions, git commit),
  registry version, step records with input and output hashes, warnings, the
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
  match, capability-set Jaccard, key-parameter accuracy, guardrail-outcome match
  and correct refusal/block rate. Each case is also reported item by item (every
  step in order, every key parameter, the issues of each action), so a live
  mismatch shows exactly what the planner chose differently; `--case <id>` runs
  one case, such as the fully specified flagship (`s1_volatility_flag_success`).
  Reports go to `eval/reports/`. In mock mode this checks the harness only
  (the mock knows just the five examples); `--mode live` measures the real model.
* **Are the computations right?** Measured by the unit tests, independently of
  any model: returns, differences, spread alignment, rolling volatility and
  annualisation, the no-look-ahead property of expanding flags, forward returns,
  group comparisons and episode counts, the Newey-West test against a direct
  statsmodels fit, every guardrail, plan editing, replay, grounding, a source
  safety scan, and a golden record of the five example workflows' results.

## Project layout

```
pyproject.toml, uv.lock     dependencies (direct ones pinned exactly)
CLAUDE.md                   engineering conventions for AI-assisted development
assets/logo.svg             the QuantLab logo (the app's query_stats icon)
docs/GUIDE.md               this guide
docs/ARCHITECTURE.md        design, module map, capability and guardrail tables
docs/SPEC.md                the v1 product specification
docs/DECISIONS.md           design decisions and their rationale
data/                       catalog.json, snapshots/*.csv (committed), README (sources, licences)
scripts/fetch_data.py       refreshes the FRED snapshot and catalog
src/quantlab/               the package (module map: ARCHITECTURE.md)
app/                        Streamlit: streamlit_app.py (router), ui.py, pages/
notebooks/01..08            one notebook per layer, executed in mock mode
eval/                       golden_requests.yaml, run_eval.py, reports/
tests/                      pytest suite (synthetic data, simulated model clients)
runs/                       run records, plan cache, events.jsonl (git-ignored)
```
