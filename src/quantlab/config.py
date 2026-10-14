"""Settings: every configurable value in one place, read from `.env` and the environment.

Role in the architecture: the engine, UI, CLI, notebooks and tests all receive a
`Settings` object instead of reading environment variables themselves. That keeps
configuration explicit (you can see exactly what a run used) and lets tests point
the whole system at temporary folders.

Environment variables (all optional):
    ANTHROPIC_API_KEY          key for the planner and narrator LLM calls
    QUANTLAB_PLANNER_MODEL     model id for the planner (default: claude-sonnet-5-5)
    QUANTLAB_NARRATOR_MODEL    model id for the narrator (default: same as planner)
    QUANTLAB_PLANNER_EFFORT    planner reasoning effort: low|medium|high (default: medium)
    QUANTLAB_NARRATOR_EFFORT   narrator reasoning effort: low|medium|high (default: low)
    QUANTLAB_LLM_MODE          live | cached | mock (default: cached with a key, mock without)
    QUANTLAB_DATA_DIR          folder holding catalog.json and snapshots/ (default: ./data)
    QUANTLAB_RUNS_DIR          folder for run records, plan cache and events (default: ./runs)
    QUANTLAB_MAX_STEPS         budget: maximum steps per plan (default: 15)
    QUANTLAB_MAX_DATASETS      budget: maximum distinct datasets per plan (default: 5)
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

DEFAULT_MODEL = "claude-sonnet-5-5"


@dataclass(frozen=True)
class Settings:
    """All settings for one process. Frozen so nothing changes them mid-run.

    Fields:
        anthropic_api_key: API key, or None when not configured.
        planner_model: model id used by the planner.
        narrator_model: model id used by the narrator.
        llm_mode: default LLM mode ("live", "cached" or "mock").
        data_dir: folder with catalog.json and the snapshots/ folder.
        runs_dir: folder for run records, the plan cache and the event log.
        max_steps: budget, maximum number of steps in a plan.
        max_datasets: budget, maximum number of distinct datasets in a plan.
        max_request_chars: budget, longest request accepted (protects the prompt size).
        planner_effort: reasoning effort for the planner call.
        narrator_effort: reasoning effort for the narrator call.
        llm_timeout_seconds: timeout for one LLM API call.
        llm_max_retries: SDK retries on rate limits / server errors (kept low for live demos).
        planner_max_tokens: output token cap for the planner call (includes thinking).
        narrator_max_tokens: output token cap for the narrator call (includes thinking).
    """

    anthropic_api_key: str | None
    planner_model: str
    narrator_model: str
    llm_mode: str
    data_dir: Path
    runs_dir: Path
    max_steps: int = 15
    max_datasets: int = 5
    max_request_chars: int = 2000
    planner_effort: str = "medium"
    narrator_effort: str = "low"
    llm_timeout_seconds: float = 90.0
    llm_max_retries: int = 1
    planner_max_tokens: int = 16000
    narrator_max_tokens: int = 8000

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
    def has_api_key(self) -> bool:
        """True when an Anthropic API key is configured."""
        return bool(self.anthropic_api_key)


def load_settings() -> Settings:
    """Build Settings from `.env` (if present) and environment variables.

    Real environment variables win over `.env` values (`override=False`), which
    is the usual convention and lets tests or CI override anything.

    Returns:
        A frozen Settings object.

    Raises:
        ValueError: if QUANTLAB_LLM_MODE is set to an unknown mode.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    # Treat an empty string (as in the .env.example template) the same as "not set".
    api_key = os.environ.get("ANTHROPIC_API_KEY") or None
    planner_model = os.environ.get("QUANTLAB_PLANNER_MODEL") or DEFAULT_MODEL
    narrator_model = os.environ.get("QUANTLAB_NARRATOR_MODEL") or planner_model

    # Without a key the only mode that can work is mock, so make it the default.
    default_mode = "cached" if api_key else "mock"
    llm_mode = os.environ.get("QUANTLAB_LLM_MODE") or default_mode
    if llm_mode not in LLM_MODES:
        raise ValueError(f"QUANTLAB_LLM_MODE must be one of {LLM_MODES}, got {llm_mode!r}.")

    return Settings(
        anthropic_api_key=api_key,
        planner_model=planner_model,
        narrator_model=narrator_model,
        llm_mode=llm_mode,
        data_dir=Path(os.environ.get("QUANTLAB_DATA_DIR") or PROJECT_ROOT / "data"),
        runs_dir=Path(os.environ.get("QUANTLAB_RUNS_DIR") or PROJECT_ROOT / "runs"),
        max_steps=int(os.environ.get("QUANTLAB_MAX_STEPS") or 15),
        max_datasets=int(os.environ.get("QUANTLAB_MAX_DATASETS") or 5),
        planner_effort=os.environ.get("QUANTLAB_PLANNER_EFFORT") or "medium",
        narrator_effort=os.environ.get("QUANTLAB_NARRATOR_EFFORT") or "low",
    )
