"""Settings: every configurable value in one place, read from `.env` and the environment.

Role in the architecture: the engine, UI, CLI, notebooks and tests all receive a
`Settings` object instead of reading environment variables themselves. That keeps
configuration explicit (you can see exactly what a run used) and lets tests point
the whole system at temporary folders.

Environment variables (all optional):
    QUANTLAB_LLM_PROVIDER      openai | anthropic: which LLM API to call (default: openai)
    OPENAI_API_KEY             key used when the provider is openai
    ANTHROPIC_API_KEY          key used when the provider is anthropic
    QUANTLAB_PLANNER_MODEL     planner model id (default: gpt-6-luna / claude-sonnet-5-5)
    QUANTLAB_NARRATOR_MODEL    model id for the narrator (default: same as planner)
    QUANTLAB_PLANNER_EFFORT    planner reasoning effort: low|medium|high (default: medium)
    QUANTLAB_NARRATOR_EFFORT   narrator reasoning effort: low|medium|high (default: low)
    QUANTLAB_LLM_MODE          live | cached | mock (default: cached with a key, mock without)
    QUANTLAB_DATA_DIR          folder holding catalog.json and snapshots/ (default: ./data)
    QUANTLAB_RUNS_DIR          folder for run records, plan cache and events (default: ./runs)
    QUANTLAB_MAX_STEPS         budget: maximum steps per plan (default: 15)
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# src/quantlab/config.py -> parents[0] is src/quantlab, [1] is src, [2] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# live:   always call the LLM (and refresh the plan cache).
# cached: reuse a cached plan for an identical request if one exists, else call the LLM.
# mock:   never call an LLM; use hand-written plans and the template narrator.
LLM_MODES = ("live", "cached", "mock")

# Which LLM API the planner and narrator call in live/cached mode. Both providers
# get the same prompts, the same plan schema and the same validation.
LLM_PROVIDERS = ("openai", "anthropic")

# Default models (the narrator uses the planner's model unless set separately).
# OpenAI: gpt-6-luna, the cost-efficient GPT-6 model, used live for both calls;
# gpt-6.1-sol can be selected with QUANTLAB_PLANNER_MODEL as an alternative.
DEFAULT_MODELS = {"openai": "gpt-6-luna", "anthropic": "claude-sonnet-5-5"}

# LLM prices in US dollars per million tokens, by model id. Used only to show
# costs on the Monitoring page, computed from the token counts the API reported.
# GPT-6 Luna: OpenAI API pricing (https://openai.com/api/pricing), checked on
# 2026-10-18. Update the numbers here when prices change. A model without an
# entry has no cost (shown as n/a). Discounts (cached input, batch) are not
# modelled, so a cost is an upper bound.
LLM_PRICES_PER_MILLION_TOKENS = {"gpt-6-luna": {"input": 0.10, "output": 0.50}}


@dataclass(frozen=True)
class Settings:
    """All settings for one process. Frozen so nothing changes them mid-run.

    Fields:
        anthropic_api_key: Anthropic API key, or None when not configured.
        planner_model: model id used by the planner.
        narrator_model: model id used by the narrator.
        llm_mode: default LLM mode ("live", "cached" or "mock").
        data_dir: folder with catalog.json and the snapshots/ folder.
        runs_dir: folder for run records, the plan cache and the event log.
        max_steps: budget, maximum number of steps in a plan.
        planner_effort: reasoning effort for the planner call.
        narrator_effort: reasoning effort for the narrator call.
        llm_timeout_seconds: timeout for one LLM API call.
        llm_max_retries: SDK retries on rate limits / server errors (kept low so a stuck
            call fails fast and the user can switch to cached or mock mode).
        planner_max_tokens: output token cap for the planner call (includes thinking).
        narrator_max_tokens: output token cap for the narrator call (includes thinking).
        llm_provider: "openai" or "anthropic".
        openai_api_key: OpenAI API key, or None when not configured.
    """

    anthropic_api_key: str | None
    planner_model: str
    narrator_model: str
    llm_mode: str
    data_dir: Path
    runs_dir: Path
    max_steps: int = 15
    planner_effort: str = "medium"
    narrator_effort: str = "low"
    llm_timeout_seconds: float = 120.0
    llm_max_retries: int = 1
    planner_max_tokens: int = 16000
    narrator_max_tokens: int = 8000
    llm_provider: str = "openai"
    openai_api_key: str | None = None

    @property
    def catalog_path(self) -> Path:
        """Path of the dataset catalog file."""
        return self.data_dir / "catalog.json"

    @property
    def cache_dir(self) -> Path:
        """Folder where planner outputs are cached."""
        return self.runs_dir / "cache"

    @property
    def events_path(self) -> Path:
        """Append-only monitoring event log (one JSON object per line)."""
        return self.runs_dir / "events.jsonl"

    @property
    def api_key(self) -> str | None:
        """The API key of the configured provider (None if not set)."""
        return self.openai_api_key if self.llm_provider == "openai" else self.anthropic_api_key

    @property
    def api_key_name(self) -> str:
        """Name of the environment variable holding the configured provider's key."""
        return "OPENAI_API_KEY" if self.llm_provider == "openai" else "ANTHROPIC_API_KEY"

    @property
    def has_api_key(self) -> bool:
        """True when the configured provider's API key is set."""
        return bool(self.api_key)


def load_settings() -> Settings:
    """Build Settings from `.env` (if present) and environment variables.

    Real environment variables win over `.env` values (`override=False`), which
    is the usual convention and lets tests or CI override anything.

    Returns:
        A frozen Settings object.

    Raises:
        ValueError: if QUANTLAB_LLM_PROVIDER or QUANTLAB_LLM_MODE has an unknown value.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    provider = os.environ.get("QUANTLAB_LLM_PROVIDER") or "openai"
    if provider not in LLM_PROVIDERS:
        raise ValueError(f"QUANTLAB_LLM_PROVIDER must be one of {LLM_PROVIDERS}, got {provider!r}.")

    # Treat an empty string (as in the .env.example template) the same as "not set".
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY") or None
    openai_key = os.environ.get("OPENAI_API_KEY") or None
    provider_key = openai_key if provider == "openai" else anthropic_key
    planner_model = os.environ.get("QUANTLAB_PLANNER_MODEL") or DEFAULT_MODELS[provider]
    narrator_model = os.environ.get("QUANTLAB_NARRATOR_MODEL") or planner_model

    # Without a key the only mode that can work is mock, so make it the default.
    default_mode = "cached" if provider_key else "mock"
    llm_mode = os.environ.get("QUANTLAB_LLM_MODE") or default_mode
    if llm_mode not in LLM_MODES:
        raise ValueError(f"QUANTLAB_LLM_MODE must be one of {LLM_MODES}, got {llm_mode!r}.")

    return Settings(
        llm_provider=provider,
        openai_api_key=openai_key,
        anthropic_api_key=anthropic_key,
        planner_model=planner_model,
        narrator_model=narrator_model,
        llm_mode=llm_mode,
        data_dir=Path(os.environ.get("QUANTLAB_DATA_DIR") or PROJECT_ROOT / "data"),
        runs_dir=Path(os.environ.get("QUANTLAB_RUNS_DIR") or PROJECT_ROOT / "runs"),
        max_steps=int(os.environ.get("QUANTLAB_MAX_STEPS") or 15),
        planner_effort=os.environ.get("QUANTLAB_PLANNER_EFFORT") or "medium",
        narrator_effort=os.environ.get("QUANTLAB_NARRATOR_EFFORT") or "low",
        # OpenAI's reasoning guide recommends reserving at least 25,000 output
        # tokens (reasoning + answer); Anthropic's SDK advises ~16,000 for
        # non-streaming calls to stay under its HTTP timeout.
        planner_max_tokens=25000 if provider == "openai" else 16000,
    )
