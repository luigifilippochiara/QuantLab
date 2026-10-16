"""Settings: provider selection, default models and modes, budgets from the environment."""

import pytest

from quantlab.config import DEFAULT_MODELS, load_settings

# Every variable load_settings reads. Setting them to "" in a test means "not set"
# and stops a developer's real .env from leaking in (dotenv never overrides them).
VARIABLES = [
    "QUANTLAB_LLM_PROVIDER",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "QUANTLAB_PLANNER_MODEL",
    "QUANTLAB_NARRATOR_MODEL",
    "QUANTLAB_LLM_MODE",
]


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in VARIABLES:
        monkeypatch.setenv(name, "")
    return monkeypatch


def test_defaults_to_openai_in_mock_mode_without_a_key(clean_env: pytest.MonkeyPatch) -> None:
    settings = load_settings()
    assert settings.llm_provider == "openai"
    assert settings.planner_model == DEFAULT_MODELS["openai"] == settings.narrator_model
    assert DEFAULT_MODELS["openai"] == "gpt-6-luna"
    assert settings.llm_mode == "mock" and not settings.has_api_key
    assert settings.planner_max_tokens == 25000  # OpenAI: room for reasoning tokens


def test_openai_key_enables_cached_mode(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("OPENAI_API_KEY", "sk-test")
    settings = load_settings()
    assert settings.has_api_key and settings.api_key_name == "OPENAI_API_KEY"
    assert settings.llm_mode == "cached"


def test_anthropic_provider_uses_its_own_key_and_model(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("QUANTLAB_LLM_PROVIDER", "anthropic")
    clean_env.setenv("OPENAI_API_KEY", "sk-test")  # the other provider's key does not count
    settings = load_settings()
    assert settings.planner_model == DEFAULT_MODELS["anthropic"]
    assert not settings.has_api_key and settings.llm_mode == "mock"
    assert settings.planner_max_tokens == 16000


def test_model_override_and_bad_values(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("QUANTLAB_PLANNER_MODEL", "gpt-6.1-sol")
    assert load_settings().narrator_model == "gpt-6.1-sol"  # narrator follows the planner
    clean_env.setenv("QUANTLAB_LLM_PROVIDER", "gemini")
    with pytest.raises(ValueError, match="QUANTLAB_LLM_PROVIDER"):
        load_settings()
