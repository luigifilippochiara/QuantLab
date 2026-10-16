# CLAUDE.md: working on QuantLab

QuantLab turns natural-language requests into quantitative analyses using only
approved, pre-written capabilities. Read `README.md` for the overview,
`docs/SPEC.md` for the original specification and `docs/DECISIONS.md` for every
decision the spec did not cover. Record new decisions there (one short
paragraph: what, why, alternatives).

## Core principle (non-negotiable)

The LLM proposes, the system disposes.

1. The planner LLM writes a plan: inert JSON naming registered capabilities.
2. `guardrails.py` checks it (G01-G14, G11 retired: BLOCK / APPROVAL / WARN).
3. A human reviews, edits parameter VALUES only (structure locked in the engine,
   re-validated by the same validator, approvals reset) and approves.
4. `executor.py` runs it by dict lookup in `REGISTRY`, with runtime guardrails R01-R04.
5. The narrator LLM summarises using only the facts sheet; `check_grounding` verifies every number.
6. Every run is saved (`lineage.py`) and can be replayed without the LLM with matching hashes.

Hard rules:

- Never use `eval`, `exec`, `compile`, `__import__`, `importlib.import_module` or
  `getattr` in `src/` (enforced by `tests/test_safety.py`). Dispatch is a dict lookup.
- The LLM is used only in `planner.py` and `narrator.py`; only `planner.py` imports
  the provider SDKs (`openai`, `anthropic`). Both are enforced by a test. The
  provider is chosen by `QUANTLAB_LLM_PROVIDER` (default `openai`).
- The LLM never sees data values: only the generated registry summary and
  dataset metadata (`data.describe_datasets`).
- Numbers shown to the user come from the executor, never from the LLM.

## Architecture (src/quantlab/)

`config` (settings) -> `types` (enums, Artifact, hashing) -> `data` (catalog,
hash-verified snapshots) -> `capabilities` (the 11 quant functions, descriptive
only; the ONLY code calling pandas/numpy for analysis) -> `registry` (specs,
REGISTRY, LLM summary, version) -> `plan` (Pydantic Plan/Step) -> `guardrails`
(static guardrails, dry run, GUARDRAILS table) -> `mock_scenarios` (example workflows) ->
`executor` (dispatch, runtime guardrails, step records) -> `planner` /
`narrator` (the only LLM code; OpenAI and Anthropic) -> `lineage` (manifest, replay, trace) ->
`monitoring` (events.jsonl) -> `engine` (Session orchestrator used by app, CLI
and notebooks) -> `evaluation`, `cli`. The Streamlit app is in `app/`
(user interface only).

## Adding a capability

Keep the menu small and simple: the capabilities are descriptive on purpose
(no statistical tests or forecasts yet; see `docs/DECISIONS.md`).

1. Write the function in `capabilities.py`: Artifacts in, a new Artifact out,
   metadata set explicitly, comments explaining every quantitative choice.
2. Add a `CapabilitySpec` to `registry.py` and to the `REGISTRY` list (inputs with
   allowed kinds, params with ranges/choices/defaults, min_obs rule, assumptions,
   library calls, `meta_from_params` if a parameter sets output metadata).
3. Add hand-computed numerical tests in `tests/test_capabilities.py`;
   `tests/test_registry.py` checks the signature matches the spec, and
   `tests/test_executor.py` checks the dry run matches runtime metadata.
4. The registry version changes automatically, which invalidates cached plans.

## Code style (every line must be readable and explainable by a reviewer)

- Python 3.12+, managed with uv. Direct dependencies pinned exactly in
  `pyproject.toml`; do not add libraries without a recorded decision
  (deliberately excluded: scipy, statsmodels, scikit-learn, arch, LangChain and
  other frameworks).
- Clear, explicit code: plain functions, Pydantic models (for untrusted input
  and records) and dataclasses (for specs). No metaprogramming, no
  registration decorators, no deep class hierarchies.
- Every module starts with a docstring explaining its role in the architecture;
  every function has a docstring (what it does, inputs, outputs).
- Comment non-obvious lines, and every quantitative decision with WHY (ddof=1,
  shifted expanding quantile, square-root-of-time annualisation...).
  Do not comment trivial lines.
- Type hints everywhere. Keep `uv run ruff check .` and `uv run ruff format --check .` clean.
- Tests need no network and no API key: synthetic data from `tests/conftest.py`,
  the fake OpenAI and Anthropic clients in `tests/fakes.py`.

## Commands

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run python -m quantlab.cli demo        # must report every scenario as expected
uv run python -m quantlab.cli replay <run_id>
uv run python -m quantlab.cli eval --mode mock
uv run streamlit run app/streamlit_app.py
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

Commit at the end of each meaningful piece of work with a descriptive message.
