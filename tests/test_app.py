"""Streamlit app, driven headlessly with streamlit.testing.AppTest (mock mode, synthetic data).

Covers the demo path: scenario A planned, approved, executed, narrated, traced
and replayed; a blocked scenario cannot execute; the other pages render.
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


def test_scenario_a_full_flow(app_env: Settings) -> None:
    app = open_page("analysis.py")
    app.sidebar.button(key="scenario-A").click().run()
    button(app, "Generate plan").click().run()
    assert not app.exception, app.exception

    assert button(app, "Execute").disabled  # the G12 approval is not ticked yet
    approve = next(c.key for c in app.checkbox if c.key.startswith("approve-"))
    app.checkbox(key=approve).check().run()
    button(app, "Execute").click().run()
    assert not app.exception, app.exception

    successes = [s.value for s in app.success]
    assert any("numbers verified" in text for text in successes)
    assert [h.value for h in app.subheader][-2:] == [
        "Trace: where does this number come from?",
        "Run record",
    ]
    assert len(app.get("plotly_chart")) >= 4

    button(app, "Replay this run").click().run()
    assert any("8 of 8 step outputs match" in s.value for s in app.success)


def test_blocked_scenario_cannot_execute(app_env: Settings) -> None:
    app = open_page("analysis.py")
    app.sidebar.button(key="scenario-C").click().run()
    button(app, "Generate plan").click().run()
    assert button(app, "Execute").disabled
    assert any("blocked" in m.value for m in app.markdown if "Run `" in m.value)


@pytest.mark.parametrize("page", ["run_history.py", "monitoring.py", "registry.py"])
def test_other_pages_render(page: str, app_env: Settings) -> None:
    open_page(page)
