"""Streamlit app, driven headlessly with streamlit.testing.AppTest (mock mode, synthetic data).

Covers the demo path: scenario 1 planned, approved, executed, narrated, traced
and replayed; blocked scenarios cannot execute; the other pages render.
"""

import pytest
from streamlit.testing.v1 import AppTest

from quantlab.config import PROJECT_ROOT, Settings

PAGES = PROJECT_ROOT / "app" / "pages"


@pytest.fixture
def app_env(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Point the app at the synthetic data and make `import ui` resolve."""
    monkeypatch.setenv("QUANTLAB_DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("QUANTLAB_RUNS_DIR", str(settings.runs_dir))
    monkeypatch.setenv("QUANTLAB_LLM_MODE", "mock")
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / "app"))
    return settings


def open_page(name: str) -> AppTest:
    """Load and run one page script."""
    app = AppTest.from_file(str(PAGES / name), default_timeout=60)
    app.run()
    assert not app.exception, app.exception
    return app


def button(app: AppTest, label: str):
    """The first button whose label starts with `label`."""
    return next(b for b in app.button if b.label.startswith(label))


def test_scenario_1_full_flow(app_env: Settings) -> None:
    app = open_page("analysis.py")
    labels = [b.label for b in app.sidebar.button]
    assert labels[0] == "1 · Volatility flag (success)" and len(labels) == 5
    app.sidebar.button(key="scenario-volatility_flag_success").click().run()
    button(app, "Generate plan").click().run()
    assert not app.exception, app.exception

    assert button(app, "Execute").disabled  # the G12 approval is not ticked yet
    approve = next(c.key for c in app.checkbox if c.key.startswith("approve-"))
    app.checkbox(key=approve).check().run()
    button(app, "Execute").click().run()
    assert not app.exception, app.exception

    successes = [s.value for s in app.success]
    assert any("numbers verified" in text for text in successes)
    subheaders = [h.value for h in app.subheader]
    # The trace panel sits directly under the results.
    assert subheaders[subheaders.index("Results") + 1] == "Trace a result"
    assert subheaders[-1] == "Run record"
    assert len(app.get("plotly_chart")) >= 4  # daily, weekly, returns, shaded volatility

    button(app, "Replay this run").click().run()
    assert any("5 of 5 step outputs match" in s.value for s in app.success)


@pytest.mark.parametrize(
    "scenario_id", ["random_forest_block", "prompt_injection_block", "frequency_mismatch_block"]
)
def test_blocked_scenario_cannot_execute(scenario_id: str, app_env: Settings) -> None:
    app = open_page("analysis.py")
    app.sidebar.button(key=f"scenario-{scenario_id}").click().run()
    button(app, "Generate plan").click().run()
    assert button(app, "Execute").disabled
    assert any("blocked" in m.value for m in app.markdown if "Run `" in m.value)


@pytest.mark.parametrize("page", ["run_history.py", "monitoring.py", "registry.py"])
def test_other_pages_render(page: str, app_env: Settings) -> None:
    open_page(page)
