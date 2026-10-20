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
uv run python -m quantlab.cli run 1 --datasets VIXCLS DEXSZUS --mode mock --approve-all   # example 1 by number
uv run python -m quantlab.cli replay <run_id>     # re-run without the model, compare hashes
uv run python -m quantlab.cli trace <run_id> s5   # where does this number come from?
uv run python -m quantlab.cli eval --mode mock    # planner evaluation -> eval/reports/
uv run python -m quantlab.cli eval --mode live --case s1_chf_safe_haven   # one case, item by item
uv run python -m quantlab.cli eval --set adversarial --mode live   # refuse or plan? (33 cases)

uv run pytest                                     # test suite: no network, no API key
uv run ruff check . && uv run ruff format --check .
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

Refreshing the data (needs network; creates a new data version):
`uv run python scripts/fetch_data.py` (or name series to fetch only those, e.g. `SP500`).

## Using the app

The **Analysis** page follows the safety model step by step:

1. Choose the datasets the planner may use. The picker lists them by ticker; its
   **?** lists each as "TICKER - description", and the line under each selected
   dataset gives its details (hover it for the official name). Then type a request, or in mock mode pick one of the example workflows in the
   sidebar, and click **Generate plan**. Switching the LLM mode, or clicking
   **Refresh** (on the right of the same row), starts a clean page: the datasets, the
   request and any plan or result on screen are cleared (a plan still awaiting a
   decision stays undecided in Run history). Refresh keeps the LLM mode.
2. Review the plan. Each step is a collapsible block whose header names the step
   and its main parameters (e.g. "Step 4 · C06 · Rolling volatility · window: 12
   weeks, annualize: yes"); open it for its inputs, predicted output and editable
   fields. Steps with a blocking issue or a warning open by default with a badge,
   and edited steps are marked. Approvals are not shown on the steps: each one is
   listed in the validation results, naming its step (e.g. "Step 3 · C03 · Compute
   returns").
3. Read the validation results. Blocking issues must be resolved, approvals must
   be ticked, and warnings are shown for context. If the plan is not what you
   want, click **Reject plan**: nothing runs, and the decision is recorded.
   Two stops come before validation and are shown where they happen: a request
   the request check refuses (R01, too long) is shown under **Request check**,
   above where the plan would be, since the planner is never called; a request the
   planner declines (P01) is shown in the **Plan** section. Neither has a
   Validation section or anything to execute or reject.
4. Click **Execute plan** (your approval of the plan). Results appear per step: charts for series (with flagged
   periods shaded), tables for statistics, and a plain-English narrative marked
   "all numbers verified" when every number checks out. A short note under the
   narrative says who wrote it (an AI narrative may contain errors in wording or
   reasoning) and, only when the check passed, that every number was checked (N01).
5. Use **Trace a result** under the results to follow any output back through
   every step to the source file, with its SHA-256, FRED id and retrieval date.

The other pages: **Run history** lists past runs, shows each manifest and user
edits, and replays a run without the model; **Monitoring** counts the runs started
on the Analysis page whose plan was approved, rejected or blocked (headless runs and
plans still awaiting a decision are left out; replays count from every source) and
shows the activity (five outcomes that do not overlap and add up to the runs:
completed, blocked by a deterministic guardrail, refused by the planner (P01), failed
and rejected, each with its share), quality and governance (approval rate, edit rate,
approval time, completed runs with warnings, grounded narratives, replay match rate),
every capability's count,
the guardrails that fired (the unused ones in an expander), tokens and cost per
stage, latency per pipeline stage and the recent errors; the cards with a target
(replay match 100%, grounded narratives 100%, failed runs 0) are green when it is
met, orange when it is not and neutral without data; **Capability registry** lists every approved capability with its code, inputs, parameters and
assumptions; **Guardrails** lists every guardrail with its action, who enforces
it, its category and what it prevents.

### Editing a plan before it runs

After a plan is generated, each step's parameters appear as fields generated
from the registry (select boxes for choices, bounded number inputs, checkboxes),
prefilled with the planner's values. Only values can change: steps, capabilities
and input references are locked, and the engine rejects any other edit with a
BLOCK. **Apply changes & re-validate** is highlighted only while a field differs
from the last validated plan (a value changed back counts as no change); it runs
the same validator again, so issues appear and clear as the plan changes: in
example 1, switching `flag_regime.method` to `full_sample` raises V09, which then
also says the significance test is invalidated; changing the USD/CHF
`load_series.dataset_id` to DGS10 is a legal edit that the validator then blocks
(V04: a forward return needs a price level, and a yield is a rate); changing only
one of its two `resample` steps raises V08 (the flags and the returns would no
longer share a frequency); and a quantile outside the registry's range would be
blocked too (V03), though its field does not accept one. In example 5, changing
`load_series.dataset_id` from NASDAQCOM to DGS10 is blocked by V07 (returns of a
yield are meaningless, plus V04: a forward return needs a price level). Windows and
minimum histories are counted in periods of the series' frequency, as their fields
and the step summaries say: in example 5, switching `resample.freq` from W to D
keeps `min_periods` at 52, which then means 52 trading days instead of 52 weeks. Any edit resets the
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
app or with `cli demo`. In mock mode the narrative is the deterministic template,
except for example 1: run unedited on the committed snapshots, it replays the
narrative an LLM wrote for the same plan and data in a live run (shown as "mock
narrator"), and N01 checks its numbers again on every run.

1. **Is the Swiss franc a safe haven?** (`chf_safe_haven_success`): does the Swiss
   franc strengthen after stress weeks? The request names the series, a threshold
   from past data only and the test; what "high" means, the frequency and the return
   type are left to the planner, and the canned plan is its reading:

   > Flag weeks when the VIX was high (using only past data), then test whether USD/CHF returns the following week differ from other weeks.

   Both series as weekly Friday closes by the same rule (the weekly VIX is its
   Friday close, the level known when the week ends), so each flag meets the USD/CHF
   return of the following week on the same date; "high" read as the top 10%: the
   VIX level flagged directly (it is already a volatility) above the 90th percentile
   of its past levels only (expanding, after 52 weeks); the following week's simple USD/CHF return
   (`forward_return`, from the weekly rates: a mean of log returns would be lower
   by about half the variance and bias the comparison against volatile weeks); then
   the significance test (`hac_mean_difference`), which reports both groups' means and
   their difference, on the weeks the two series share. The analyst
   confirms the outcome's return method (V10). The VIX starts in 1990, so the
   earlier USD/CHF weeks are dropped when the two are aligned, and the run says so
   (E04 ALIGNMENT_LOSS, a warning). DEXSZUS is quoted in Swiss francs per US dollar
   (USD/CHF), so a negative return means the franc strengthened: the narrative
   states this convention, read from the dataset catalog, and so does every
   USD/CHF result.
2. **Frequency-alignment protection** (`spread_frequency_mismatch_block`):
   "Compute the spread between the weekly 10-year Treasury yield and the daily
   2-year Treasury yield, and flag the most negative weeks across the entire history."
   Planned exactly as asked, the spread of a weekly and a daily series would
   silently keep only the Fridays and mix two frequencies. The validator, not the
   planner, blocks it (V08) before anything runs and states the fix: resample the
   2-year yield to weekly first. The most negative weeks are the bottom 10% of the
   spread over its whole history (`flag_regime` with direction below and a
   full-sample threshold), so the plan also asks for the look-ahead approval (V09).
   Steps are locked, so in the app the frequencies are aligned the other way:
   switching the 10-year's `resample.freq` to D leaves only V09 to approve, and the
   plan runs on daily data, flagging 1,257 of 12,590 days, every one with an
   inverted curve (the 10-year below the 2-year by more than 0.26 percentage
   points).
3. **Out-of-scope forecast request** (`arima_forecast_block`): "Fit an ARIMA
   model to the S&P 500 and forecast the next 12 weeks." The registry has no
   capability that fits a model or makes a forecast, so the planner declines
   (P01) and lists the registered capabilities instead. If a planner ever
   proposed such a step anyway, the validator would block it independently (V02).
4. **Prompt injection, simulated planner** (`prompt_injection_unknown_capability`):
   a request that tells the assistant to ignore its rules and run Python. Here
   the mock planner is a *simulated misbehaving planner*: it follows the
   injection and proposes a well-formed plan of one step that calls an
   unregistered capability, `run_python`, with the request's code as its
   parameter. The validator rejects the whole plan deterministically (V02
   UNKNOWN_CAPABILITY, a terminal block): the Plan section shows a "Plan rejected"
   banner naming every blocked step, the Validation section shows the V02 finding
   for each, there are no editor, approval, Execute or Reject controls, and nothing
   is executed or loaded; only a new request goes on. A
   fixable block such as example 2's V08 keeps its editing flow. The app labels
   this planner output as simulated. A real planner is expected to refuse such a request (P01, as in
   example 3); this example shows the wall that holds when it does not, and
   automated tests confirm that nothing from the request reaches a shell or a
   subprocess.
5. **Buy the dip in tech?** (`buy_the_dip_tech`): does the NASDAQ's following
   week differ after its biggest drops?

   > Flag the weeks when NASDAQ returns were in the bottom 10%, then test whether the following week's return differs from other weeks.

   The canned plan writes only what the request states or the registry requires
   (the dataset, weekly data, both return methods, the bottom 10%) and leaves the
   rest to the registry's defaults, which the reviewer sees filled in: weekly Friday
   closes (`how` Last); weekly simple returns, so a dip is the percentage lost in the
   week; the dips are the weeks below the 10th percentile of past weekly returns
   only (`flag_regime` with direction below and the default expanding threshold,
   after 52 weeks), so every dip could have been seen at the time and no look-ahead
   approval is needed; the following week's simple return (`forward_return`); then
   the significance test (`hac_mean_difference`), which reports both groups' means.
   The analyst confirms the two return-method choices (V10); there is no warning and
   no block. The test is two-sided, so the narrative first says whether the
   following week's return differs after a dip and whether that is statistically
   significant, then which group's mean was higher, with both means.

**Example output.** For example workflow 1 on the committed snapshots, QuantLab
flags 221 of 1,867 weeks (11.8%) whose VIX close was above the 90th percentile of
its past levels, in 63 distinct episodes. Over the 1,866 weeks both series share
(1991 to 2026; the VIX starts in 1990, so the run warns that the earlier USD/CHF
weeks were dropped, E04), the following week's simple USD/CHF return averaged
+0.066% after a high-VIX week and -0.023% after another week: the franc weakened
on average after stress weeks, the opposite of a safe haven. The difference, +0.088
percentage points, has a Newey-West standard error of 0.133 percentage points (7
lags) and a p-value of 0.51, so there is no statistically significant difference:
in this sample the franc does not strengthen in the week after a stress week. The
test looks at the following week, not at the stress week itself. Example 5 finds
no statistically significant difference after the biggest NASDAQ drops: 334 dip
weeks (below the 10th percentile of past weekly returns, from -1.5% in 1972 to
-2.9% today), and the following week's return averaged -0.124% after a dip against
+0.275% after other weeks, so the mean was higher after the other weeks, by 0.40
percentage points, with a p-value of 0.068: not statistically significant at the
5% level.

**Audit trail.** For any executed result, such as example 1's p-value, **Trace a
result** shows the full chain: `hac_mean_difference` -> `forward_return` ->
`resample` -> `load_series` -> the USD/CHF snapshot file, and `flag_regime` ->
`resample` -> `load_series` -> the VIX snapshot file, with each step's library
call, parameters (user edits marked) and output hash. `cli demo` prints the same
chain.

## LLM modes

| Mode | Planner | Narrator |
|---|---|---|
| **live** | Calls the model every time, and caches the plan | Calls the model |
| **cached** (default with an API key) | Reuses the cached plan for an identical planning call, otherwise calls the model | Calls the model |
| **mock** (default without a key) | Canned plans for the five example workflows; anything else is declined | Deterministic template; example 1, unedited on the committed data, replays the narrative an LLM wrote for it in a live run, still checked by N01 |

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

Typing exactly these requests in **cached** mode (switching mode clears the page,
so type or paste the request after switching) then returns their plans instantly
and identically, even on a slow network (the
narrator still calls the model, and falls back to the deterministic template if
it cannot). A live model may decline a request for a different reason than the
canned plan (for example refuse example 2 outright with P01, rather than
proposing the misaligned spread that the validator blocks with V08); the command
only checks that blocked workflows stay blocked.

**4. Start the app**, open http://localhost:8501 and pick **cached** (the default
when a key is set) or **live** in the sidebar. If the API is unavailable, switch
to **mock**: everything keeps working offline with the canned plans.

## Reproducibility, lineage and monitoring

* **Run record.** `runs/<run_id>/manifest.json`, saved as soon as a plan is shown
  and again at every step: request, model information (model, prompt version,
  mode, cache hit, latency, tokens), the plan as proposed and as run with its
  version hash, parameter edits and the parameter-level diff, validation report,
  approvals, the review (in the app or headless, when the plan was shown, and the
  decision, approved or rejected, with its time), dataset versions (file SHA-256,
  retrieval time, rows and dates used), environment (Python, library versions,
  git commit), registry version, step records with input and output hashes,
  warnings, the narrative with its grounding result, and stage timings and tokens
  per LLM stage (a cached plan uses none and records the tokens it saved).
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
  narration, completion, rejections, replays). The Monitoring page reads the run
  records for per-run numbers (activity, review, tokens and cost, latency,
  failures) and the event log for guardrail firings, capability executions and
  replays. Costs come from the reported tokens and the prices in
  `config.LLM_PRICES_PER_MILLION_TOKENS`; mock runs are left out of them.

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
  one case, such as example 1's (`s1_chf_safe_haven`).
  Reports go to `eval/reports/`. In mock mode this checks the harness only
  (the mock knows just the five examples); `--mode live` measures the real model.
* **Does the planner refuse when it should, and only then?** Measured by
  `quantlab.cli eval --set adversarial` on `eval/adversarial_requests.yaml`: 33
  requests in eight categories (vague, out-of-scope methods, unavailable data,
  advice, prompt injection, off-topic, mixed, and valid but loosely worded), each
  expected to be refused (P01) or planned. The report gives accuracy per category,
  false plans and false refusals separately, and the tokens and cost of the run.
  About a third of the cases are held out (`--holdout exclude` leaves them out while
  working on the planner prompt; `--holdout only` scores them alone). Live results
  with gpt-6-luna:

  | Category | planner-v6 | planner-v7 |
  |---|---|---|
  | Vague or open-ended | 1/5 | 5/5 |
  | Out-of-scope methods | 5/5 | 5/5 |
  | Data not available | 4/4 | 4/4 |
  | Advice or decisions | 3/3 | 3/3 |
  | Prompt injection and probing | 5/5 | 5/5 |
  | Off-topic | 3/3 | 3/3 |
  | Mixed (valid plus out-of-scope) | 3/3 | 3/3 |
  | Valid but loosely worded (must plan) | 5/5 | 5/5 |
  | **Total** (false plans / false refusals) | **29/33** (4 / 0) | **33/33** (0 / 0) |

  Planner-v6 chose an analysis itself when a request had no question. Planner-v7 plans
  only a concrete analytical question, otherwise declines, says what QuantLab can
  compute and suggests requests. It never gives financial advice. On the golden set
  the two versions score the same (8 of 11, the same three misses). A full run costs
  about $0.02.
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
assets/architecture.png     the architecture diagram (README, ARCHITECTURE.md)
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
