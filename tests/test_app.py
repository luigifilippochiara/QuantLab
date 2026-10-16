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


def issue_codes(app: AppTest) -> set[str]:
    """Guardrail codes shown in the validation panel (from the issue headers)."""
    return {m.value.split("**")[1].split()[0] for m in app.markdown if "badge[" in m.value
            and "**" in m.value}  # fmt: skip


def plan_scenario_1(app_env: Settings) -> AppTest:
    """The analysis page with scenario 1's plan generated."""
    app = open_page("analysis.py")
    app.sidebar.button(key="scenario-volatility_flag_success").click().run()
    button(app, "Generate plan").click().run()
    return app


def test_weekly_to_daily_edit_warns_and_runs_after_re_approval(app_env: Settings) -> None:
    app = plan_scenario_1(app_env)
    assert any("adjust parameters below" in i.value for i in app.info)
    window = next(n for n in app.number_input if n.key.endswith("-s4-window"))
    assert window.label == "window (in periods of the series frequency)" and window.value == 12

    next(s for s in app.selectbox if s.key.endswith("-s2-freq")).select("D").run()
    assert button(app, "Execute").disabled  # the change is not applied yet
    assert any(c.value == "Apply or reset your parameter changes to continue." for c in app.caption)

    button(app, "Apply changes").click().run()
    assert issue_codes(app) == {"G12", "G14"}
    assert any("Approvals were reset" in i.value for i in app.info)
    assert any("edited (was W)" in c.value for c in app.caption)

    app.checkbox(key=next(c.key for c in app.checkbox if c.key.startswith("approve-"))).check()
    app.run()
    button(app, "Execute").click().run()
    assert not app.exception, app.exception
    trace = next(d.value for d in app.dataframe if "library call" in d.value.columns)
    params = dict(zip(trace["step"], trace["params"], strict=True))
    assert "freq=D (edited, was W)" in params["s2"]
    assert "(edited" not in params["s4"]  # planner-proposed values are not marked


def test_full_sample_edit_raises_g10_and_reset_clears_it(app_env: Settings) -> None:
    app = plan_scenario_1(app_env)
    next(s for s in app.selectbox if s.key.endswith("-s5-method")).select("full_sample").run()
    button(app, "Apply changes").click().run()
    assert issue_codes(app) == {"G10", "G12"}
    button(app, "Reset to planner").click().run()
    assert issue_codes(app) == {"G12"}
