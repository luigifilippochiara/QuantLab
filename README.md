# QuantLab

QuantLab turns natural-language requests for quantitative analysis into a
**structured, inspectable execution plan**, which is **validated** and then
**executed by a deterministic engine**.

The key idea: the LLM *proposes* a plan, written as data (JSON). It never writes
or runs code. Everything after the plan is plain, tested Python, so the same
plan on the same data always gives the same result, and every result can be
traced back to the plan that produced it.

## Architecture

```
  "Show me 20-day volatility"        (natural language)
              |
              v
        +-----------+
        |  planner  |   LLM proposes a plan using only registered capabilities   [not built yet]
        +-----------+
              |   plan (JSON): steps -> capability name + params + input/output names
              v
        +------------+
        | validation |   is every capability registered? params typed and in range?
        +------------+   inputs exist with the right data type?
              |   valid plan                 (errors can go back to the planner)
              v
        +-----------+
        | execution |   looks each capability up in an allow-list and runs the
        +-----------+   matching pre-written function, step by step
              |
              v
           results  ------>  monitoring: audit trail of request, plan, run   [not built yet]

  registry: the single source of truth for what capabilities exist and which
            parameters they accept. Read by planner, validation and execution.
  data:     loading and checking input data.                                  [not built yet]
```

| Module | Role | Status |
|---|---|---|
| `core/registry.py` | Declares capabilities: parameters (type + allowed range), input/output data types, description | One capability: `rolling_volatility` |
| `core/validation.py` | Plan schema (`Plan`, `PlanStep`) and checks against the registry; returns all errors at once | First version |
| `core/execution.py` | Capability implementations + dispatch that refuses anything not registered; runs a plan step by step | First version |
| `core/planner.py` | Natural language -> plan, via an LLM | Stub |
| `core/data.py` | Load and check input data | Stub |
| `core/monitoring.py` | Audit log of requests, plans and runs | Stub |

## Safety model

- **No code from the LLM.** A plan can only *name* a capability. The engine maps
  that name to a pre-written function through a fixed dict; anything else is refused.
- **Two layers of checks.** `validate_plan` checks the whole plan up front and
  lists every problem. `run_capability` re-checks each call at the point of use,
  so skipping validation doesn't open a back door.
- **No hidden defaults.** Every parameter must be written in the plan, so the
  plan alone fully describes the computation.
- **Typed data flow.** Each capability declares input and output data types
  (e.g. `price_series` -> `volatility_series`), so multi-step plans can be
  type-checked before running.

## Plan format

```json
{
  "plan_id": "example-rolling-vol-001",
  "description": "20-day annualised rolling volatility of the sample daily closing prices.",
  "steps": [
    {
      "step_id": "s1",
      "capability": "rolling_volatility",
      "input": "prices",
      "params": {"window": 20, "periods_per_year": 252},
      "output": "vol_20d"
    }
  ]
}
```

`input` names a dataset supplied by the caller (or an earlier step's `output`);
`output` names where this step's result is stored.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                   # create .venv and install dependencies
uv run python scripts/run_example.py      # validate + execute the example plan
```

Optional:

```bash
uv run python scripts/make_sample_data.py  # regenerate the sample prices (same seed -> same file)
cp .env.example .env                       # add an API key (only needed once the planner exists)
```

## Project layout

```
quantlab/
├── core/                  the engine (see table above)
├── data/sample/           small synthetic price series, committed so the demo runs
├── plans/                 hand-written example plans (JSON)
├── scripts/               runnable entry points
├── notebooks/             exploration notebooks
├── pyproject.toml         project metadata and dependencies (managed by uv)
└── .env.example           template for secrets; copy to .env (git-ignored)
```

## Sample data

`data/sample/sample_prices.csv` is **simulated**, not real market data: one year
of daily prices with a known annual volatility of 20% (fixed random seed). This
gives a built-in sanity check: computed volatility should scatter around 0.20.
With a 20-day window the scatter is wide, since 20 returns is a small sample.

## Next steps

1. Planner: give the LLM the registry and the plan schema, ask for a plan as
   structured JSON, and feed validation errors back for a corrected plan.
2. More capabilities (returns, drawdown, correlation, ...) to make multi-step plans useful.
3. Data layer with input checks (sorted dates, duplicates, missing values).
4. Monitoring: log each request -> plan -> validation -> result.
5. Tests (pytest) for each capability and for the validator.
