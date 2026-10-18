"""Streamlit app, driven headlessly with streamlit.testing.AppTest (mock mode, synthetic data).

Covers the demo path: scenario 1 planned, approved, executed, narrated, traced
and replayed; blocked scenarios cannot execute; every issue shows who enforces it;
an ungrounded LLM narrative is withheld (N01); the other pages render; the
header, mock-only example workflows and full dataset names on hover.
"""

import dataclasses
import json

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from quantlab.config import PROJECT_ROOT, Settings
from quantlab.data import load_catalog
from quantlab.engine import Session, run_request
from quantlab.guardrails import ENFORCERS, GUARDRAILS, Action
from quantlab.lineage import LLMInfo, RunRecord, load_runs, new_run_id, save_run
from quantlab.mock_scenarios import FLAGSHIP_REQUEST, SCENARIOS_BY_ID
from quantlab.monitoring import guardrail_counts, read_events, utc_now
from quantlab.narrator import NUMBERS_CHECKED_NOTE
from quantlab.plan import Step
from quantlab.registry import REGISTRY
from quantlab.types import TokenUsage

from conftest import SCENARIO_1_APPROVALS
from fakes import FakeAnthropicClient, text_response

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
    # The plan is reviewed in the readable view only: no raw JSON of it on this page.
    assert not any("JSON" in e.label for e in app.expander) and not app.json

    assert button(app, "Execute").disabled  # the V10 approvals are not ticked yet
    approve_all(app)
    button(app, "Execute").click().run()
    assert not app.exception, app.exception

    successes = [s.value for s in app.success]
    assert any("numbers verified" in text for text in successes)
    # The note directly under the (template) narrative: its numbers were checked.
    captions = [c.value for c in app.caption]
    assert NUMBERS_CHECKED_NOTE in captions
    # Results and trace name each capability by code and name.
    results = [e.label for e in app.expander if e.label.startswith("s")]
    assert results[3].startswith("s4 · C06 · rolling_volatility · completed")
    assert results[7].startswith("s8 · C10 · hac_mean_difference · completed")
    # The significance test's table, every number formatted as in the facts sheet.
    test = next(
        d.value for d in app.dataframe
        if "statistic" in d.value.columns and "p_value" in d.value["statistic"].tolist()
    )  # fmt: skip
    for statistic in ["diff_means", "se_hac", "p_value", "n_episodes_flagged", "maxlags_used"]:
        assert statistic in test["statistic"].tolist()
    # The trace starts at the final result, the test with its p-value.
    picker = next(b for b in app.selectbox if b.label == "Step to trace")
    assert picker.value == "s8" and picker.options[-1] == "s8 · C10 · hac_mean_difference"
    trace = next(d.value for d in app.dataframe if "library call" in d.value.columns)
    assert trace[["step", "code", "capability"]].values.tolist()[:3] == [
        ["s8", "C10", "hac_mean_difference"],
        ["s6", "C08", "forward_return"],
        ["s5", "C07", "flag_regime"],
    ]
    subheaders = [h.value for h in app.subheader]
    # The narrative (with its note) comes right after the results, then the trace.
    assert subheaders[-4:] == ["Results", "Narrative", "Trace a result", "Run record"]
    trace_intro = next(i for i, c in enumerate(captions) if c.startswith("Pick a step to see"))
    assert captions.index(NUMBERS_CHECKED_NOTE) < trace_intro
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
    "scenario_id",
    ["arima_forecast_block", "prompt_injection_block", "spread_frequency_mismatch_block"],
)
def test_blocked_scenario_cannot_execute(scenario_id: str, app_env: Settings) -> None:
    app = open_page("analysis.py")
    app.sidebar.button(key=f"scenario-{scenario_id}").click().run()
    button(app, "Generate plan").click().run()
    assert button(app, "Execute").disabled
    assert any("blocked" in m.value for m in app.markdown if "Run `" in m.value)
    # A blocked plan can still be rejected; a refusal holds no plan to reject.
    refusal = scenario_id != "spread_frequency_mismatch_block"
    assert button(app, "Reject plan").disabled is refusal


def test_the_reviewer_can_reject_a_plan(app_env: Settings) -> None:
    app = open_page("analysis.py")
    app.sidebar.button(key="scenario-volatility_flag_success").click().run()
    button(app, "Generate plan").click().run()
    button(app, "Reject plan").click().run()
    assert not app.exception, app.exception
    assert button(app, "Execute").disabled and button(app, "Reject plan").disabled
    assert any(i.value.startswith("You rejected this plan") for i in app.info)
    [record] = load_runs(app_env.runs_dir)
    assert (record.status, record.decision, record.interactive) == ("rejected", "rejected", True)


@pytest.mark.parametrize(
    ("scenario_id", "enforcer", "category"),
    [
        ("volatility_flag_success", "Validator · Type: Deterministic", "Methodology"),
        ("arima_forecast_block", "Planner · Type: LLM", "Approved scope"),
        ("spread_frequency_mismatch_block", "Validator · Type: Deterministic", "Methodology"),
    ],
)
def test_every_issue_shows_who_enforces_it(
    scenario_id: str, enforcer: str, category: str, app_env: Settings
) -> None:
    app = open_page("analysis.py")
    app.sidebar.button(key=f"scenario-{scenario_id}").click().run()
    button(app, "Generate plan").click().run()
    enforced_by, kind = enforcer.split(" · ")
    shown = f"Enforced by: **{enforced_by}** · {kind} · Category: {category}"
    caption = next(c for c in app.caption if c.value == shown)
    # The guardrail's description is the line's tooltip, from the same GUARDRAILS table.
    assert caption.proto.help in {g.description for g in GUARDRAILS.values()}


def test_ungrounded_llm_narrative_is_withheld_with_an_n01_issue(app_env: Settings) -> None:
    # Planned and executed in mock mode; only the narrative comes from a (fake) LLM,
    # which invents a number.
    scenario = SCENARIOS_BY_ID["volatility_flag_success"]
    session = Session.start(scenario.request, list(scenario.dataset_ids), "mock", app_env)
    session.create_plan()
    session.approve(SCENARIO_1_APPROVALS)
    session.execute()
    invented = "Volatility peaked at 123.456 during the flagged weeks."
    session.mode, session.client = "live", FakeAnthropicClient(text_response(invented))
    session.narrate()
    session.save_record()

    app = AppTest.from_file(str(PAGES / "analysis.py"), default_timeout=60)
    app.session_state["session"] = session
    app.run()
    assert not app.exception, app.exception
    captions = [c.value for c in app.caption]
    n01 = "Enforced by: **Narrator check** · Type: Deterministic · Category: Output grounding"
    assert n01 in captions
    assert any("N01 UNGROUNDED_NARRATIVE" in m.value and "narrative" in m.value
               for m in app.markdown)  # fmt: skip
    assert not any("numbers verified" in s.value for s in app.success)
    assert not any("has been checked" in c for c in captions)  # never claimed after N01
    # The withheld text appears once, in a collapsed expander, with its failing number.
    shown = invented.replace(".", "\\.")  # as escaped for Markdown (ui.escape_md)
    assert [m.value for m in app.markdown if shown in m.value] == [shown]
    withheld = next(e for e in app.expander if e.label.startswith("Withheld LLM narrative"))
    assert [m.value for m in withheld.markdown] == [shown] and not withheld.proto.expanded
    assert "123.456" in " ".join(c.value for c in withheld.caption)
    assert any("Written by: deterministic template" in c for c in captions)


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


def guardrail_codes(app: AppTest) -> list[str]:
    """Codes of the rows the Guardrails page's table shows ([] when no row matches)."""
    return list(app.table[0].value.index) if app.table else []


def showing(app: AppTest) -> str:
    """The "Showing n of N guardrails" count above the table."""
    return next(c.value for c in app.caption if c.value.startswith("Showing "))


def test_guardrails_page_intro_and_table_come_from_the_code(app_env: Settings) -> None:
    app = open_page("guardrails.py")
    assert app.title[0].value == "Guardrails"
    intro = app.caption[0].value
    for action in Action:
        assert f"**{action.value}**" in intro  # what each action means
    table = app.table[0].value
    assert list(table.index) == list(GUARDRAILS)
    assert list(table.columns) == [
        "Name", "Action", "Enforced by", "Type", "Category", "Description"]  # fmt: skip
    v09 = table.loc["V09"]
    assert (v09["Action"], v09["Enforced by"], v09["Type"], v09["Category"]) == (
        "APPROVE", "Validator", "Deterministic", "Methodology")  # fmt: skip
    assert table.loc["P01", "Type"] == "LLM"
    assert table.loc["V02", "Description"] == GUARDRAILS["V02"].description
    assert showing(app) == f"Showing {len(GUARDRAILS)} of {len(GUARDRAILS)} guardrails"
    assert button(app, "Clear filters").disabled  # nothing to clear yet


def test_guardrail_filter_options_come_from_the_table(app_env: Settings) -> None:
    app = open_page("guardrails.py")
    guardrails = list(GUARDRAILS.values())
    options = {m.key: m.options for m in app.multiselect}
    assert options["guardrail_action"] == list(dict.fromkeys(g.action.value for g in guardrails))
    assert options["guardrail_enforcer"] == list(dict.fromkeys(g.enforced_by for g in guardrails))
    assert options["guardrail_enforcer"] == list(ENFORCERS.values())  # every stage, in order
    assert options["guardrail_type"] == ["Deterministic", "LLM"]  # GuardrailType order
    assert set(options["guardrail_category"]) == {g.category.value for g in guardrails}


@pytest.mark.parametrize(
    ("search", "expected"),
    [
        ("V0", [f"V0{i}" for i in range(1, 10)]),  # partial code
        ("R0", ["R01"]),  # the request check only: no executor check is labelled R
        ("request", ["R01"]),  # by name
        ("look", ["V09"]),  # partial name, any case
        ("p01", ["P01"]),  # code, any case
        ("  data ", ["V06", "V11", "E01", "E02", "E03"]),  # names with "DATA", spaces ignored
        ("zzz", []),
    ],
)
def test_guardrail_search_matches_code_or_name(
    search: str, expected: list[str], app_env: Settings
) -> None:
    app = open_page("guardrails.py")
    app.text_input(key="guardrail_search").input(search).run()
    assert guardrail_codes(app) == expected
    assert showing(app) == f"Showing {len(expected)} of {len(GUARDRAILS)} guardrails"
    if not expected:
        assert app.info[0].value == "No guardrail matches these filters."


def test_guardrail_filters_combine_and_clear(app_env: Settings) -> None:
    app = open_page("guardrails.py")
    app.multiselect(key="guardrail_action").select("APPROVE").run()
    assert guardrail_codes(app) == ["V09", "V10"]
    app.multiselect(key="guardrail_action").unselect("APPROVE").run()
    app.multiselect(key="guardrail_enforcer").select("Executor").run()
    assert guardrail_codes(app) == ["E01", "E02", "E03", "E04", "E05"]
    app.multiselect(key="guardrail_category").select("Methodology").run()  # AND
    assert guardrail_codes(app) == ["E05"]
    app.multiselect(key="guardrail_enforcer").select("Planner").run()  # OR within one
    app.multiselect(key="guardrail_category").select("Approved scope").run()
    assert guardrail_codes(app) == ["P01", "E05"]
    app.text_input(key="guardrail_search").input("e0").run()
    assert guardrail_codes(app) == ["E05"]
    assert showing(app) == f"Showing 1 of {len(GUARDRAILS)} guardrails"

    button(app, "Clear filters").click().run()
    assert guardrail_codes(app) == list(GUARDRAILS)
    assert app.text_input(key="guardrail_search").value == ""
    assert all(m.value == [] for m in app.multiselect)
    assert button(app, "Clear filters").disabled


def test_guardrail_type_filter_alone_and_combined(app_env: Settings) -> None:
    app = open_page("guardrails.py")
    app.multiselect(key="guardrail_type").select("LLM").run()
    assert guardrail_codes(app) == ["P01"]
    assert showing(app) == f"Showing 1 of {len(GUARDRAILS)} guardrails"
    app.multiselect(key="guardrail_action").select("WARN").run()  # AND: no LLM warning
    assert guardrail_codes(app) == []
    app.multiselect(key="guardrail_type").unselect("LLM").select("Deterministic").run()
    assert guardrail_codes(app) == ["E03", "E04", "E05", "N01"]
    app.multiselect(key="guardrail_category").select("Data quality").run()
    app.text_input(key="guardrail_search").input("loss").run()
    assert guardrail_codes(app) == ["E04"]
    button(app, "Clear filters").click().run()
    assert guardrail_codes(app) == list(GUARDRAILS)
    assert app.multiselect(key="guardrail_type").value == []


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


def test_weekly_to_daily_edit_re_validates_and_runs_after_re_approval(app_env: Settings) -> None:
    app = plan_scenario_1(app_env)
    assert any("adjust parameters below" in i.value for i in app.info)
    window = next(n for n in app.number_input if n.key.endswith("-s4-window"))
    assert window.label == "window (in periods of the series frequency)" and window.value == 12

    next(s for s in app.selectbox if s.key.endswith("-s2-freq")).select("D").run()
    assert button(app, "Execute").disabled  # the change is not applied yet
    assert any(c.value == "Apply or reset your parameter changes to continue." for c in app.caption)

    button(app, "Apply changes").click().run()
    assert issue_codes(app) == {"V10"}  # the return-method approvals again, nothing else
    assert any("Approvals were reset" in i.value for i in app.info)
    assert any("edited (was W)" in c.value for c in app.caption)

    approve_all(app)
    button(app, "Execute").click().run()
    assert not app.exception, app.exception
    trace = next(d.value for d in app.dataframe if "library call" in d.value.columns)
    params = dict(zip(trace["step"], trace["params"], strict=True))
    assert "freq=D (edited, was W)" in params["s2"]
    assert "(edited" not in params["s4"]  # planner-proposed values are not marked


def test_full_sample_edit_raises_v09_and_reset_clears_it(app_env: Settings) -> None:
    app = plan_scenario_1(app_env)
    next(s for s in app.selectbox if s.key.endswith("-s5-method")).select("full_sample").run()
    button(app, "Apply changes").click().run()
    assert issue_codes(app) == {"V09", "V10"}
    button(app, "Reset to planner").click().run()
    assert issue_codes(app) == {"V10"}


def step_headers(app: AppTest) -> list[tuple[str, bool]]:
    """(label, open) of every plan-step expander, in plan order."""
    return [(e.label, e.proto.expanded) for e in app.expander if e.label.startswith("Step ")]


def test_plan_steps_are_collapsed_and_approvals_stay_in_validation(app_env: Settings) -> None:
    app = plan_scenario_1(app_env)
    headers = step_headers(app)
    assert [is_open for _, is_open in headers] == [False] * 8  # the V10 steps too
    assert headers[0][0] == "Step 1 · C01 · Load series · NASDAQCOM — Synthetic equity index"
    assert headers[2][0] == "Step 3 · C03 · Compute returns · method: log, periods: 1"  # no badge
    assert headers[3][0] == "Step 4 · C06 · Rolling volatility · window: 12 weeks, annualize: yes"
    assert headers[5][0] == "Step 6 · C08 · Forward return · method: simple"
    assert headers[7][0] == "Step 8 · C10 · HAC mean difference"  # no parameters: no summary
    # The V10 approvals are still required, in the Validation section, naming their steps.
    v10 = [m.value for m in app.markdown if "**V10 MATERIAL_ASSUMPTION**" in m.value]
    assert v10[0].endswith("· Step 3 · C03 · Compute returns (s3)")
    assert v10[1].endswith("· Step 6 · C08 · Forward return (s6)")
    assert len([c for c in app.checkbox if c.key.startswith("approve-")]) == 2
    assert button(app, "Execute").disabled
    # Collapsing is visual only: the editor still runs.
    assert next(n for n in app.number_input if n.key.endswith("-s4-window")).value == 12


def test_edits_are_marked_and_window_units_follow_the_frequency(app_env: Settings) -> None:
    app = plan_scenario_1(app_env)
    next(s for s in app.selectbox if s.key.endswith("-s2-freq")).select("D").run()
    button(app, "Apply changes").click().run()
    headers = step_headers(app)
    assert headers[1] == ("Step 2 · C02 · Resample · freq: D, how: last :blue-badge[:material/"
                          "edit: edited]", False)  # fmt: skip
    # Window 12 now counts trading days: the summary says so, and nothing warns.
    assert headers[3] == (
        "Step 4 · C06 · Rolling volatility · window: 12 trading days, annualize: yes",
        False,
    )
    assert not any("Warning]" in label for label, _ in headers)
    # The new V10 approvals of this plan version open nothing and add no badge.
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
    header = ui.step_header(2, unknown, "steps ahead: 1", {Action.BLOCK}, edited=False)
    assert header == "Step 2 · Arima · steps ahead: 1 :red-badge[:material/block: Blocked]"


def chart_bars(app: AppTest) -> list[int]:
    """The number of bars (or points) in each plotly chart of the page, in order."""
    return [
        sum(len(trace.get("y", [])) for trace in json.loads(chart.proto.spec)["data"])
        for chart in app.get("plotly_chart")
    ]


def test_monitoring_page_with_no_runs_shows_a_note(app_env: Settings) -> None:
    app = open_page("monitoring.py")
    assert app.info[0].value.startswith("No runs yet")
    assert not app.get("plotly_chart") and not app.metric and not app.subheader


def test_monitoring_page_sections_cards_and_charts(app_env: Settings) -> None:
    scenario = SCENARIOS_BY_ID["volatility_flag_success"]
    # 1. Approved and executed on the Analysis page; 2. rejected there.
    for decide in ("approve", "reject"):
        session = Session.start(scenario.request, list(scenario.dataset_ids), "mock", app_env,
                                interactive=True)  # fmt: skip
        session.create_plan()
        if decide == "reject":
            session.reject()
            continue
        session.approve([i.issue_id for i in session.unresolved_approvals()])
        session.execute()
        session.narrate()
        session.save_record()
    # 3. A headless run blocked by V08; 4. a failed run; 5. a live run with tokens.
    blocked = SCENARIOS_BY_ID["spread_frequency_mismatch_block"]
    run_request(blocked.request, list(blocked.dataset_ids), "mock", settings=app_env)
    now = utc_now()
    save_run(
        RunRecord(
            run_id=new_run_id(),
            created_at=now,
            status="failed",
            request="x",
            error="boom",
            stage_timings_ms={"planning": 5.0},
        ),
        {},
        app_env.runs_dir,
    )
    luna = {"model": "gpt-6-luna"}
    save_run(RunRecord(run_id=new_run_id() + "x", created_at=now, status="completed",
                       request="y", llm=LLMInfo(provider="openai", prompt_version="planner-v6",
                                                mode="live", **luna),
                       stage_timings_ms={"planning": 1200.0, "narration": 800.0},
                       stage_tokens={"planning": TokenUsage(input_tokens=1500, output_tokens=400,
                                                            **luna),
                                     "narration": TokenUsage(input_tokens=1000,
                                                             output_tokens=200, **luna)}),
             {}, app_env.runs_dir)  # fmt: skip

    app = open_page("monitoring.py")
    assert [h.value for h in app.subheader] == ["Activity", "Quality and governance",
        "Guardrails", "Capabilities", "Cost", "Latency", "Recent errors"]  # fmt: skip
    assert not app.date_input and not app.multiselect  # no filters: every run is shown
    cards = {m.proto.label: m.proto for m in app.metric}
    values = {
        label: cards[label].body
        for label in ("Total runs", "Completed", "Blocked", "Failed", "Rejected by human")
    }
    assert values == {"Total runs": "5", "Completed": "2 (40%)", "Blocked": "1 (20%)",
                      "Failed": "1 (20%)", "Rejected by human": "1 (20%)"}  # fmt: skip
    # Failed runs need attention: the card turns amber; a clean N01 rate does not.
    assert "Needs attention" in cards["Failed"].delta and cards["Failed"].color == 3  # ORANGE
    assert "Needs attention" not in cards["N01 failure rate"].delta
    # Two plans reviewed in the app: one approved, one rejected; the headless run is left out.
    assert cards["Approval rate"].body == "50%"
    assert cards["Approval rate"].delta == "1 approved · 1 rejected · 0 abandoned"
    # The guardrails that fired; all of them (in the expander); every capability; cost per
    # stage; mean and p95 for each stage with data (all but the untimed request check).
    fired = guardrail_counts(read_events(app_env.events_path), include_unused=False)
    assert chart_bars(app) == [len(fired), len(GUARDRAILS), len(REGISTRY), 2, 2 * 5]
    every = next(e for e in app.expander if e.label == "Show all guardrails (including unused)")
    assert chart_bars(every) == [len(GUARDRAILS)]
    cost, latency = [t.value for t in app.table]
    errors = app.dataframe[0].value
    assert cost.loc["planner", "Cost, US$ (mean)"] == pytest.approx(0.00035)
    assert cost.loc["total", "Input tokens (mean)"] == 2500
    assert latency.index.tolist() == [
        "Request check", "Planner", "Validator", "Approval", "Executor", "Narrator"]  # fmt: skip
    assert pd.isna(latency.loc["Request check", "Mean (ms)"])  # not timed: shown as a dash
    # Dollar signs in a caption are escaped, or a pair of them would render as LaTeX.
    (prices,) = [c.value for c in app.caption if c.value.startswith("Per run")]
    assert r"Total \$0.000550" in prices and r"gpt-6-luna \$0.10 input" in prices
    assert errors["code"].tolist() == ["—", "V08"] and errors["message"][0] == "boom"


def test_monitoring_page_before_any_guardrail_fired(app_env: Settings) -> None:
    save_run(RunRecord(run_id=new_run_id(), created_at=utc_now(), status="completed",
                       request="x"), {}, app_env.runs_dir)  # fmt: skip
    app = open_page("monitoring.py")
    assert any(c.value == "No guardrail has fired yet." for c in app.caption)
    # The expander still lists every guardrail, at zero.
    every = next(e for e in app.expander if e.label == "Show all guardrails (including unused)")
    assert chart_bars(every) == [len(GUARDRAILS)]
