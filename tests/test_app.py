"""Streamlit app, driven headlessly with streamlit.testing.AppTest (mock mode, synthetic data).

Covers the demo path: scenario 1 planned, approved, executed, narrated, traced
and replayed; blocked scenarios cannot execute; the other pages render; the
header, mock-only example workflows and full dataset names on hover.
"""

import dataclasses
import datetime

import pytest
from streamlit.testing.v1 import AppTest

from quantlab.config import PROJECT_ROOT, Settings
from quantlab.data import load_catalog
from quantlab.guardrails import GUARDRAILS, Severity
from quantlab.mock_scenarios import CANNED_REQUESTS_BY_ID, FLAGSHIP_REQUEST, SCENARIOS_BY_ID
from quantlab.plan import Step
from quantlab.registry import REGISTRY

PAGES = PROJECT_ROOT / "app" / "pages"
SCENARIO_1_REQUEST = SCENARIOS_BY_ID["volatility_flag_success"].request


@pytest.fixture
def app_env(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Point the app at the synthetic data and make `import ui` resolve."""
    monkeypatch.setenv("QUANTLAB_DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("QUANTLAB_RUNS_DIR", str(settings.runs_dir))
    monkeypatch.setenv("QUANTLAB_LLM_MODE", "mock")
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / "app"))
    return settings


def open_page(name: str, llm_mode: str | None = None) -> AppTest:
    """Load and run one page script, optionally with the sidebar's LLM mode preset."""
    app = AppTest.from_file(str(PAGES / name), default_timeout=60)
    if llm_mode is not None:
        app.session_state["llm_mode"] = llm_mode
    app.run()
    assert not app.exception, app.exception
    return app


def button(app: AppTest, label: str):
    """The first button whose label starts with `label`."""
    return next(b for b in app.button if b.label.startswith(label))


def approve_all(app: AppTest) -> None:
    """Tick every approval checkbox (scenario 1 has two: the return method of s3 and s6)."""
    for key in [c.key for c in app.checkbox if c.key.startswith("approve-")]:
        app.checkbox(key=key).check()
    app.run()


def test_scenario_1_full_flow(app_env: Settings) -> None:
    app = open_page("analysis.py")
    labels = [b.label for b in app.sidebar.button]
    assert labels[0] == "1 · High-volatility weeks and what follows" and len(labels) == 5
    app.sidebar.button(key="scenario-volatility_flag_success").click().run()
    button(app, "Generate plan").click().run()
    assert not app.exception, app.exception

    assert button(app, "Execute").disabled  # the G12 approvals are not ticked yet
    approve_all(app)
    button(app, "Execute").click().run()
    assert not app.exception, app.exception

    successes = [s.value for s in app.success]
    assert any("numbers verified" in text for text in successes)
    # Results and trace name each capability by code and name.
    results = [e.label for e in app.expander if e.label.startswith("s")]
    assert results[3].startswith("s4 · C06 · rolling_volatility · completed")
    assert results[7].startswith("s8 · C13 · hac_mean_difference · completed")
    # The significance test's table, every number formatted as in the facts sheet.
    test = next(
        d.value for d in app.dataframe
        if "statistic" in d.value.columns and "p_value" in d.value["statistic"].tolist()
    )  # fmt: skip
    for statistic in ["diff_means", "se_hac", "p_value", "n_episodes_flagged", "maxlags_used"]:
        assert statistic in test["statistic"].tolist()
    # The trace starts at the final result, the test with its p-value.
    picker = next(b for b in app.selectbox if b.label == "Step to trace")
    assert picker.value == "s8" and picker.options[-1] == "s8 · C13 · hac_mean_difference"
    trace = next(d.value for d in app.dataframe if "library call" in d.value.columns)
    assert trace[["step", "code", "capability"]].values.tolist()[:3] == [
        ["s8", "C13", "hac_mean_difference"],
        ["s6", "C12", "forward_return"],
        ["s5", "C07", "flag_regime"],
    ]
    subheaders = [h.value for h in app.subheader]
    # The trace panel sits directly under the results.
    assert subheaders[subheaders.index("Results") + 1] == "Trace a result"
    assert subheaders[-1] == "Run record"
    assert len(app.get("plotly_chart")) >= 5  # daily, weekly, returns, volatility, forward

    button(app, "Replay this run").click().run()
    assert any("8 of 8 step outputs match" in s.value for s in app.success)


def test_example_1_button_fills_in_the_flagship_request(app_env: Settings) -> None:
    app = open_page("analysis.py")
    app.sidebar.button(key="scenario-volatility_flag_success").click().run()
    assert app.text_area(key="request_text").value == FLAGSHIP_REQUEST


def test_header_describes_the_product(app_env: Settings) -> None:
    app = open_page("analysis.py")
    assert app.title[0].value.endswith("QuantLab")
    assert any("validated, fully traceable" in m.value for m in app.markdown)
    assert not any("badge[" in m.value or "replayed exactly" in m.value for m in app.markdown)
    assert any(c.value.startswith("**How it works:**") for c in app.caption)
    assert not any("Audit trail" in c.value for c in app.sidebar.caption)


@pytest.mark.parametrize("mode", ["live", "cached"])
def test_example_workflows_only_in_mock_mode(mode: str, app_env: Settings) -> None:
    assert len(open_page("analysis.py").sidebar.button) == 5  # mock: the five examples
    app = open_page("analysis.py", llm_mode=mode)
    assert len(app.sidebar.button) == 0
    assert not any(h.value == "Example workflows" for h in app.sidebar.header)
    assert not any("example workflow" in i.value for i in app.info)


def test_no_dataset_is_selected_when_the_app_opens(app_env: Settings) -> None:
    app = open_page("analysis.py")
    assert app.sidebar.multiselect[0].value == []
    assert not any("title=" in c.value for c in app.sidebar.caption)  # no dataset lines
    app.sidebar.button(key="scenario-volatility_flag_success").click().run()
    assert app.sidebar.multiselect[0].value == ["NASDAQCOM"]  # an example fills in its own


def test_datasets_are_labelled_id_first_with_full_names_on_hover(app_env: Settings) -> None:
    app = open_page("analysis.py")
    # Synthetic catalog (conftest): title "Synthetic equity index", full name with ", full name".
    datasets = app.sidebar.multiselect[0]
    assert "NASDAQCOM — Synthetic equity index" in datasets.options
    assert "- VIXCLS — Synthetic VIX, full name" in datasets.help
    datasets.select("NASDAQCOM").run()
    (line,) = [c.value for c in app.sidebar.caption if "title=" in c.value]
    assert 'title="NASDAQCOM — Synthetic equity index, full name"' in line
    assert "<b>NASDAQCOM — Synthetic equity index</b>" in line

    app.text_area(key="request_text").input(SCENARIO_1_REQUEST).run()
    button(app, "Generate plan").click().run()
    load = next(s for s in app.selectbox if s.key.endswith("-s1-dataset_id"))
    assert load.value == "NASDAQCOM"  # the plan still holds the id
    assert "NASDAQCOM — Synthetic equity index" in load.options
    assert "- SP500 — Synthetic S&P 500, full name" in load.help

    approve_all(app)
    button(app, "Execute").click().run()
    assert any(c.value == "Source: NASDAQCOM — Synthetic equity index" for c in app.caption)
    sources = next(d.value for d in app.dataframe if "file sha256" in d.value.columns)
    assert sources["dataset"].tolist() == ["NASDAQCOM — Synthetic equity index"]
    button(app, "Replay this run").click().run()
    checks = next(d.value for d in app.dataframe if "recorded sha256" in d.value.columns)
    assert checks["dataset"].tolist() == ["NASDAQCOM — Synthetic equity index"]

    history = open_page("run_history.py")
    assert any(c.value == "Datasets: NASDAQCOM — Synthetic equity index" for c in history.caption)


@pytest.mark.parametrize(
    "scenario_id", ["arima_forecast_block", "prompt_injection_block", "frequency_mismatch_block"]
)
def test_blocked_scenario_cannot_execute(scenario_id: str, app_env: Settings) -> None:
    app = open_page("analysis.py")
    app.sidebar.button(key=f"scenario-{scenario_id}").click().run()
    button(app, "Generate plan").click().run()
    assert button(app, "Execute").disabled
    assert any("blocked" in m.value for m in app.markdown if "Run `" in m.value)


@pytest.mark.parametrize(
    "page", ["run_history.py", "monitoring.py", "registry.py", "guardrails.py"]
)
def test_other_pages_render(page: str, app_env: Settings) -> None:
    open_page(page)


def test_capability_count_comes_from_the_registry(app_env: Settings) -> None:
    app = open_page("registry.py")
    assert f"these {len(REGISTRY)} capabilities" in app.caption[0].value
    # The guardrails moved to their own page: shown in one place only.
    assert not app.table and [h.value for h in app.subheader] == []


def test_guardrails_page_counts_and_table_come_from_the_code(app_env: Settings) -> None:
    app = open_page("guardrails.py")
    assert app.title[0].value == "Guardrails"
    static = sum(code.startswith("G") for code in GUARDRAILS)  # plan-level G codes
    runtime = sum(code.startswith("R") for code in GUARDRAILS)  # runtime R codes
    intro = app.caption[0].value
    assert f"every plan passes {static} static guardrails" in intro
    assert f"and {runtime} runtime guardrails" in intro
    table = app.table[0].value
    assert list(table.index) == list(GUARDRAILS)
    assert list(table.columns) == ["name", "severity", "stage", "what goes wrong without it"]


def test_sidebar_title_sits_above_the_page_menu(app_env: Settings) -> None:
    """Through the router: the title first, then the five page links, then the page's own."""
    app = AppTest.from_file(str(PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=60)
    app.run()
    assert not app.exception, app.exception
    headers = [h.value for h in app.sidebar.header]
    assert headers[0] == "QuantLab"
    assert headers[1:] == ["Datasets", "LLM mode", "Example workflows"]  # Analysis, mock mode
    links = [(link.value, link.proto.icon) for link in app.sidebar.get("page_link")]
    assert links == [
        ("Analysis", ":material/query_stats:"),
        ("Run history", ":material/history:"),
        ("Monitoring", ":material/monitoring:"),
        ("Capability registry", ":material/menu_book:"),
        ("Guardrails", ":material/shield:"),
    ]
    # Drawing order: the title, then the menu, then the page's own sidebar content.
    order = [(node.type, node.value) for node in list(app.sidebar.children.values())[:7]]
    assert [kind for kind, _ in order] == ["header"] + ["page_link"] * 5 + ["header"]
    assert (order[0][1], order[6][1]) == ("QuantLab", "Datasets")


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

    approve_all(app)
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


def test_period_editing_with_date_widgets_and_extremes_table(app_env: Settings) -> None:
    app = open_page("analysis.py")
    request = CANNED_REQUESTS_BY_ID["nasdaq_volatility_peak_since_2020"].request
    app.text_area(key="request_text").input(request).run()
    button(app, "Generate plan").click().run()
    # Nullable dates: a "Set" checkbox, then a date input.
    assert [(c.label, c.value) for c in app.checkbox if c.key.endswith("-set")] == [
        ("Set start", True), ("Set end", False),
    ]  # fmt: skip
    (start,) = app.date_input
    assert start.label == "start" and start.value == datetime.date(2020, 1, 1)

    next(c for c in app.checkbox if c.key.endswith("end-set")).check().run()
    next(d for d in app.date_input if d.key.endswith("-end")).set_value(
        datetime.date(2019, 6, 30)
    ).run()
    button(app, "Apply changes").click().run()
    assert issue_codes(app) == {"G03", "G12"}  # the end is before the start

    next(c for c in app.checkbox if c.key.endswith("end-set")).uncheck().run()
    next(d for d in app.date_input if d.key.endswith("-start")).set_value(
        datetime.date(2021, 1, 1)
    ).run()
    button(app, "Apply changes").click().run()
    assert issue_codes(app) == {"G12"}

    app.checkbox(key=next(c.key for c in app.checkbox if c.key.startswith("approve-"))).check()
    app.run()
    button(app, "Execute").click().run()
    assert not app.exception, app.exception
    table = next(d.value for d in app.dataframe if "extreme" in d.value.columns)
    assert list(table.columns) == ["extreme", "date", "value", "tie_count"]
    assert table["extreme"].tolist() == ["max"] and table["date"].iloc[0] >= "2021-01-01"
    trace = next(d.value for d in app.dataframe if "library call" in d.value.columns)
    params = dict(zip(trace["step"], trace["params"], strict=True))
    assert "start=2021-01-01 (edited, was 2020-01-01)" in params["s5"]


@pytest.mark.parametrize(("edited", "rows"), [("both", ["max", "min"]), ("min", ["min"])])
def test_find_extremes_which_edit_in_the_editor(
    edited: str, rows: list[str], app_env: Settings
) -> None:
    app = open_page("analysis.py")
    # Its plan ends in find_extremes(which="max") at s6, after a G12 approval.
    request = CANNED_REQUESTS_BY_ID["nasdaq_volatility_peak_since_2020"].request
    app.text_area(key="request_text").input(request).run()
    button(app, "Generate plan").click().run()
    next(s for s in app.selectbox if s.key.endswith("-s6-which")).select(edited).run()
    button(app, "Apply changes").click().run()
    app.checkbox(key=next(c.key for c in app.checkbox if c.key.startswith("approve-"))).check()
    app.run()
    button(app, "Execute").click().run()
    assert not app.exception, app.exception
    table = next(d.value for d in app.dataframe if "extreme" in d.value.columns)
    assert table["extreme"].tolist() == rows
    assert any("numbers verified" in s.value for s in app.success)


def step_headers(app: AppTest) -> list[tuple[str, bool]]:
    """(label, open) of every plan-step expander, in plan order."""
    return [(e.label, e.proto.expanded) for e in app.expander if e.label.startswith("Step ")]


def test_plan_steps_are_collapsed_and_approvals_stay_in_validation(app_env: Settings) -> None:
    app = plan_scenario_1(app_env)
    headers = step_headers(app)
    assert [is_open for _, is_open in headers] == [False] * 8  # the G12 steps too
    assert headers[0][0] == "Step 1 · C01 · Load series · NASDAQCOM — Synthetic equity index"
    assert headers[2][0] == "Step 3 · C03 · Compute returns · method: log, periods: 1"  # no badge
    assert headers[3][0] == "Step 4 · C06 · Rolling volatility · window: 12 weeks, annualize: yes"
    assert headers[5][0] == "Step 6 · C12 · Forward return · horizon: 1 week, method: simple"
    assert headers[7][0] == "Step 8 · C13 · HAC mean difference"  # maxlags unset: no summary
    # The G12 approvals are still required, in the Validation section, naming their steps.
    g12 = [m.value for m in app.markdown if "**G12 MATERIAL_ASSUMPTION**" in m.value]
    assert g12[0].endswith("· Step 3 · C03 · Compute returns (s3)")
    assert g12[1].endswith("· Step 6 · C12 · Forward return (s6)")
    assert len([c for c in app.checkbox if c.key.startswith("approve-")]) == 2
    assert button(app, "Execute").disabled
    # Collapsing is visual only: the editor still runs.
    assert next(n for n in app.number_input if n.key.endswith("-s4-window")).value == 12


def test_edits_are_marked_and_steps_with_new_findings_open(app_env: Settings) -> None:
    app = plan_scenario_1(app_env)
    next(s for s in app.selectbox if s.key.endswith("-s2-freq")).select("D").run()
    button(app, "Apply changes").click().run()
    headers = step_headers(app)
    assert headers[1] == ("Step 2 · C02 · Resample · freq: D, how: last :blue-badge[:material/"
                          "edit: edited]", False)  # fmt: skip
    # Window 12 now counts trading days: G14 warns on both steps, which open.
    assert headers[3][1] and headers[3][0].startswith(
        "Step 4 · C06 · Rolling volatility · window: 12 trading days, annualize: yes"
    )
    assert headers[4][1] and "Warning]" in headers[3][0] and "Warning]" in headers[4][0]
    # The forward horizon is counted in periods too: 1 week became 1 trading day.
    assert headers[5][1] and headers[5][0].startswith(
        "Step 6 · C12 · Forward return · horizon: 1 trading day, method: simple"
    )
    # The new G12 approvals of this plan version open nothing and add no badge.
    assert headers[2] == ("Step 3 · C03 · Compute returns · method: log, periods: 1", False)


def test_step_summary_falls_back_to_every_parameter(app_env: Settings) -> None:
    import ui  # importable once app_env has put app/ on the path

    catalog = load_catalog(app_env.catalog_path)
    flag = Step(id="s5", capability="flag_regime", inputs={"series": "s4"},
                params={"quantile": 0.9, "method": "expanding", "min_periods": 52})  # fmt: skip
    spec = REGISTRY["flag_regime"]
    assert ui.step_summary(flag, spec, "W", catalog) == "quantile: 0.9, method: expanding"
    every = dataclasses.replace(spec, summary_params=())
    assert ui.step_summary(flag, every, "W", catalog).endswith("min periods: 52 weeks")
    unknown = Step(id="s2", capability="arima", params={"steps_ahead": 1})
    assert ui.step_summary(unknown, None, None, catalog) == "steps ahead: 1"
    header = ui.step_header(2, unknown, "steps ahead: 1", {Severity.BLOCK}, edited=False)
    assert header == "Step 2 · Arima · steps ahead: 1 :red-badge[:material/block: Blocked]"
