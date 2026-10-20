"""Streamlit app, driven headlessly with streamlit.testing.AppTest (mock mode, synthetic data).

Covers the demo path: scenario 1 planned, approved, executed, narrated, traced
and replayed; blocked scenarios cannot execute; every issue shows who enforces it;
an ungrounded LLM narrative is withheld (N01); the other pages render; the
header, mock-only example workflows and full dataset names on hover.
"""

import ast
import dataclasses
import json
import re
import tomllib

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from quantlab.config import PROJECT_ROOT, Settings
from quantlab.data import load_catalog
from quantlab.graph import build_graph, resume_run, run_graph, start_run
from quantlab.guardrails import (
    CATEGORY_DEFINITIONS,
    ENFORCERS,
    GUARDRAILS,
    MAX_REQUEST_WORDS,
    Action,
    make_issue,
)
from quantlab.lineage import LLMInfo, RunRecord, load_run, load_runs, new_run_id, replay, save_run
from quantlab.mock_scenarios import SCENARIOS_BY_ID, VIX_CHF_NARRATIVE, VIX_CHF_REQUEST
from quantlab.monitoring import (
    guardrail_counts,
    monitored_events,
    monitored_runs,
    read_events,
    utc_now,
)
from quantlab.narrator import NUMBERS_CHECKED_NOTE
from quantlab.plan import Step
from quantlab.registry import REGISTRY
from quantlab.types import TokenUsage

from conftest import (
    CHANGE_REQUEST_COMMENT,
    SCENARIO_5_APPROVALS,
    as_plan,
    execute,
    paused_ids,
    spread_weekly_plan,
    start_review,
    volatility_plan,
)
from fakes import FakeAnthropicClient, text_response, tool_response

PAGES = PROJECT_ROOT / "app" / "pages"
# labels.PERIODS_NOTE, as the analyst reads it (app/ is importable only inside a test).
PERIODS_NOTE = "Measured in periods of the series' frequency (e.g. weeks for weekly data)."
SCENARIO_5_REQUEST = SCENARIOS_BY_ID["buy_the_dip_tech"].request


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
    """Tick every approval checkbox (e.g. scenario 5's two return methods)."""
    for key in [c.key for c in app.checkbox if c.key.startswith("approve-")]:
        app.checkbox(key=key).check()
    app.run()


def test_scenario_1_full_flow(app_env: Settings) -> None:
    app = open_page("analysis.py")
    labels = [b.label for b in app.sidebar.button]
    assert labels[0] == "1 · Is the Swiss franc a safe haven?" and len(labels) == 5
    app.sidebar.button(key="scenario-chf_safe_haven_success").click().run()
    button(app, "Generate plan").click().run()
    assert not app.exception, app.exception
    # The plan is reviewed in the readable view only: no raw JSON of it on this page.
    assert not any("JSON" in e.label for e in app.expander) and not app.json

    assert button(app, "Execute plan").disabled  # the V10 approvals are not ticked yet
    approve_all(app)
    button(app, "Execute plan").click().run()
    assert not app.exception, app.exception

    successes = [s.value for s in app.success]
    assert any("numbers verified" in text for text in successes)
    # The note directly under the (template) narrative: its numbers were checked.
    captions = [c.value for c in app.caption]
    assert NUMBERS_CHECKED_NOTE in captions
    # Results and trace name each capability by code and name.
    results = [e.label for e in app.expander if e.label.startswith("s")]
    assert results[2].startswith("s3 · C07 · Flag regime · completed")
    assert results[6].startswith("s7 · C10 · HAC mean difference · completed")
    # Under each USD/CHF result, and in the narrative: which way the rate reads.
    quote = "DEXSZUS is quoted as USD/CHF, in Swiss francs per US dollar"
    assert sum(c.value.startswith(quote) for c in app.caption) == 4  # s4, s5, s6, s7
    assert any(quote in m.value for m in app.markdown)
    # The significance test's table, with both groups' means, every number formatted as
    # in the facts sheet.
    test = next(
        d.value for d in app.dataframe
        if "statistic" in d.value.columns and "p-value" in d.value["statistic"].tolist()
    )  # fmt: skip
    for statistic in ["Difference in means", "Mean (flagged)", "Mean (other)",
                      "HAC standard error", "p-value", "Flagged episodes",
                      "Newey-West lags"]:  # fmt: skip
        assert statistic in test["statistic"].tolist()
    # The trace starts at the final result, the test with its p-value.
    picker = next(b for b in app.selectbox if b.label == "Step to trace")
    assert picker.value == "s7" and picker.options[-1] == "s7 · C10 · HAC mean difference"
    trace = next(d.value for d in app.dataframe if "library call" in d.value.columns)
    assert trace[["step", "code", "capability"]].values.tolist()[:3] == [
        ["s7", "C10", "HAC mean difference"],
        ["s6", "C08", "Forward return"],
        ["s5", "C02", "Resample"],
    ]
    subheaders = [h.value for h in app.subheader]
    # The narrative (with its note) comes right after the results, then the trace.
    assert subheaders[-4:] == ["Results", "Narrative", "Trace a result", "Run record"]
    trace_intro = next(i for i, c in enumerate(captions) if c.startswith("Pick a step to see"))
    assert captions.index(NUMBERS_CHECKED_NOTE) < trace_intro
    assert len(app.get("plotly_chart")) >= 5  # daily, weekly, returns, volatility, forward

    button(app, "Replay this run").click().run()
    assert any("7 of 7 step outputs match" in s.value for s in app.success)


def test_example_1_button_fills_in_its_request(app_env: Settings) -> None:
    app = open_page("analysis.py")
    app.sidebar.button(key="scenario-chf_safe_haven_success").click().run()
    assert app.text_area(key="request_text").value == VIX_CHF_REQUEST


def plan_scenario_1(app_env: Settings) -> AppTest:
    """The analysis page in mock mode with example 1's plan generated (awaiting approval)."""
    app = open_page("analysis.py", llm_mode="mock")
    app.sidebar.button(key="scenario-chf_safe_haven_success").click().run()
    button(app, "Generate plan").click().run()
    assert app.session_state["run"].state["status"] == "ready"
    return app


def test_scenario_1_shows_its_recorded_narrative_on_the_committed_data(
    app_env: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The committed FRED snapshots: the data example 1's narrative was recorded on.
    monkeypatch.setenv("QUANTLAB_DATA_DIR", str(PROJECT_ROOT / "data"))
    catalog = load_catalog(PROJECT_ROOT / "data" / "catalog.json")
    if any(catalog.entries[d].sha256 != h for d, h in VIX_CHF_NARRATIVE.dataset_sha256.items()):
        pytest.skip("the committed snapshots changed since example 1's narrative was recorded")
    app = plan_scenario_1(app_env)
    approve_all(app)
    button(app, "Execute plan").click().run()
    assert not app.exception, app.exception
    assert any(m.value == ui_escaped(VIX_CHF_NARRATIVE.text) for m in app.markdown)
    assert any("numbers verified" in s.value for s in app.success)
    captions = [c.value for c in app.caption]
    assert any(
        c.startswith("Written by: mock narrator (an LLM narrative recorded from a live run)")
        for c in captions
    )


def ui_escaped(text: str) -> str:
    """Text as the page writes it in Markdown (dollar signs and the like escaped)."""
    import ui  # importable once app_env has put app/ on the path

    return ui.escape_md(text)


def test_switching_the_llm_mode_resets_the_analysis_page(app_env: Settings) -> None:
    import ui  # importable once app_env has put app/ on the path

    app = plan_scenario_1(app_env)
    run_id = app.session_state["run"].state["run_id"]
    saved = load_run(run_id, app_env.runs_dir)
    assert app.multiselect(key="dataset_select").value == ["VIXCLS", "DEXSZUS"]
    # A change typed but not applied yet.
    next(n for n in app.number_input if n.key.endswith("-s3-quantile")).set_value(0.95).run()

    app.segmented_control(key="llm_mode").set_value("cached").run()
    assert not app.exception, app.exception
    # Every key of the page is back to its starting value, and the widgets are drawn
    # empty on the run of the switch itself, not one run later.
    assert all(app.session_state[k] == v for k, v in ui.analysis_defaults().items())
    assert app.session_state["llm_mode"] == "cached"
    assert app.text_area(key="request_text").value == ""
    assert app.multiselect(key="dataset_select").value == []
    # No plan, validation, editor, approval, results, narrative or trace on screen.
    assert not app.subheader and not app.number_input and not app.checkbox
    assert app.info[0].value.startswith("Type a request")
    # The run that was waiting for a decision is left as it was saved: undecided.
    assert load_run(run_id, app_env.runs_dir) == saved and saved.status == "ready"


def test_refresh_resets_the_analysis_page_and_keeps_the_mode(app_env: Settings) -> None:
    import ui  # importable once app_env has put app/ on the path

    app = plan_scenario_1(app_env)
    run_id = app.session_state["run"].state["run_id"]
    saved = load_run(run_id, app_env.runs_dir)

    button(app, "Refresh").click().run()
    assert not app.exception, app.exception
    # Every key of the page is back to its starting value, drawn empty on this run.
    assert all(app.session_state[k] == v for k, v in ui.analysis_defaults().items())
    assert app.session_state["llm_mode"] == "mock"
    assert app.text_area(key="request_text").value == ""
    assert app.multiselect(key="dataset_select").value == []
    assert not app.subheader and not app.number_input and not app.checkbox
    assert app.info[0].value.startswith("Type a request")
    # The run that was waiting for a decision is left as it was saved: undecided.
    assert load_run(run_id, app_env.runs_dir) == saved and saved.status == "ready"


def test_clicking_the_selected_llm_mode_keeps_the_page(app_env: Settings) -> None:
    app = plan_scenario_1(app_env)
    run_id = app.session_state["run"].state["run_id"]
    app.segmented_control(key="llm_mode").set_value("mock").run()  # already selected
    assert app.session_state["run"].state["run_id"] == run_id
    assert app.text_area(key="request_text").value == VIX_CHF_REQUEST
    assert app.multiselect(key="dataset_select").value == ["VIXCLS", "DEXSZUS"]
    assert "Plan" in [h.value for h in app.subheader]


def test_the_examples_caption_matches_their_order(app_env: Settings) -> None:
    app = open_page("analysis.py")
    assert (
        "Ready-made requests: complete analyses (1, 5), a blocked request you can fix and "
        "re-run (2), and out-of-scope or unsafe requests QuantLab stops (3, 4)."
    ) in [c.value for c in app.sidebar.caption]
    labels = [b.label for b in app.sidebar.button]
    assert labels == [
        "1 · Is the Swiss franc a safe haven?",
        "2 · Frequency-alignment protection",
        "3 · Out-of-scope forecast request",
        "4 · Prompt injection",
        "5 · Buy the dip in tech?",
    ]


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
    app.sidebar.button(key="scenario-chf_safe_haven_success").click().run()
    # An example fills in its own.
    assert app.sidebar.multiselect[0].value == ["VIXCLS", "DEXSZUS"]


def test_datasets_are_picked_by_ticker_and_listed_ticker_first(app_env: Settings) -> None:
    app = open_page("analysis.py")
    catalog = load_catalog(app_env.catalog_path)
    # Synthetic catalog (conftest): title "Synthetic equity index", full name with ", full name".
    datasets = app.sidebar.multiselect[0]
    # The list and the selected chips show tickers; the help lists every full label.
    assert datasets.options == [catalog.entries[d].fred_id for d in catalog.ids()]
    assert "- VIXCLS - Synthetic VIX" in datasets.help
    datasets.select("NASDAQCOM").run()
    assert datasets.value == ["NASDAQCOM"]
    # The line under each selected dataset: the full label, its official name on hover.
    (line,) = [c.value for c in app.sidebar.caption if "title=" in c.value]
    assert 'title="NASDAQCOM - Synthetic equity index, full name"' in line
    assert "<b>NASDAQCOM - Synthetic equity index</b>" in line

    app.text_area(key="request_text").input(SCENARIO_5_REQUEST).run()
    button(app, "Generate plan").click().run()
    # The header is Markdown: "-" is escaped there and shows as "-".
    assert step_headers(app)[0][0] == (
        "Step 1 · C01 · Load series · NASDAQCOM \\- Synthetic equity index"
    )
    load = next(s for s in app.selectbox if s.key.endswith("-s1-dataset_id"))
    assert load.value == "NASDAQCOM"  # the plan still holds the id
    assert "NASDAQCOM - Synthetic equity index" in load.options
    assert "- SP500 - Synthetic S&P 500" in load.help

    approve_all(app)
    button(app, "Execute plan").click().run()
    label = ui_escaped("NASDAQCOM - Synthetic equity index")  # captions are Markdown
    assert any(c.value == f"Source: {label}" for c in app.caption)
    sources = next(d.value for d in app.dataframe if "file sha256" in d.value.columns)
    assert sources["dataset"].tolist() == ["NASDAQCOM - Synthetic equity index"]
    assert "FRED id" not in sources.columns  # the label starts with it
    trace = next(d.value for d in app.dataframe if "library call" in d.value.columns)
    params = dict(zip(trace["step"], trace["params"], strict=True))
    assert params["s1"] == "Dataset: NASDAQCOM - Synthetic equity index"
    button(app, "Replay this run").click().run()
    checks = next(d.value for d in app.dataframe if "recorded sha256" in d.value.columns)
    assert checks["dataset"].tolist() == ["NASDAQCOM - Synthetic equity index"]

    history = open_page("run_history.py")
    assert any(c.value == f"Datasets: {label}" for c in history.caption)


def test_every_catalog_dataset_has_a_ticker_first_label_and_a_ticker(app_env: Settings) -> None:
    import labels  # importable once app_env has put app/ on the path

    # The committed catalog, the one the app ships with.
    catalog = load_catalog(PROJECT_ROOT / "data" / "catalog.json")
    for dataset_id in catalog.ids():
        entry = catalog.entries[dataset_id]
        assert labels.dataset_label(catalog, dataset_id) == f"{entry.fred_id} - {entry.title}"
        assert labels.dataset_ticker(catalog, dataset_id) == entry.fred_id
    assert labels.dataset_label(catalog, "NASDAQCOM") == "NASDAQCOM - NASDAQ Composite Index"
    assert labels.dataset_ticker(catalog, "NASDAQCOM") == "NASDAQCOM"
    # An id that is not in the catalog is shown as it is.
    assert labels.dataset_label(catalog, "BITCOIN") == labels.dataset_ticker(catalog, "BITCOIN")
    assert labels.dataset_ticker(catalog, "BITCOIN") == "BITCOIN"


# The fields of a catalog entry a dataset label is made of.
DATASET_NAME_FIELDS = {"fred_id", "full_name", "title"}


def dataset_name_reads(source: str) -> list[str]:
    """Where a source reads a dataset's name fields (st.title, a page heading, aside)."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr in DATASET_NAME_FIELDS:
            if isinstance(node.value, ast.Name) and node.value.id == "st":
                continue
            found.append(f"line {node.lineno}: .{node.attr}")
    return found


def test_only_labels_py_builds_dataset_labels() -> None:
    # Regression guard: a page that formats a dataset's name itself is what reversed
    # the labels before; only app/labels.py may read the name fields.
    assert dataset_name_reads('label = f"{entry.title} ({entry.fred_id})"') == [
        "line 1: .title", "line 1: .fred_id"]  # fmt: skip
    assert dataset_name_reads('st.title("Run history")') == []
    offenders = {
        str(path.relative_to(PROJECT_ROOT)): found
        for path in sorted((PROJECT_ROOT / "app").rglob("*.py"))
        if path.name != "labels.py"
        and (found := dataset_name_reads(path.read_text(encoding="utf-8")))
    }
    assert offenders == {}


def test_blocked_scenario_cannot_execute(app_env: Settings) -> None:
    app = open_page("analysis.py")
    app.sidebar.button(key="scenario-spread_frequency_mismatch_block").click().run()
    button(app, "Generate plan").click().run()
    assert button(app, "Execute plan").disabled
    assert any("blocked" in m.value for m in app.markdown if "Run `" in m.value)
    # A blocked plan can still be rejected.
    assert not button(app, "Reject plan").disabled


def test_a_planner_refusal_is_shown_in_the_plan_not_in_validation(app_env: Settings) -> None:
    app = open_page("analysis.py")
    app.sidebar.button(key="scenario-arima_forecast_block").click().run()
    button(app, "Generate plan").click().run()
    assert any("blocked" in m.value for m in app.markdown if "Run `" in m.value)
    # P01 is the planner's: its card is in the Plan section, under the refusal. The
    # validator never ran, so there is no Validation section and nothing to decide.
    assert [h.value for h in app.subheader] == ["Plan"]
    assert app.error[0].proto.title == "The planner declined this request"
    (card,) = [m.value for m in app.markdown if "PLANNER_REFUSED" in m.value]
    assert card.endswith("**P01 PLANNER_REFUSED** · request")
    labels = [b.label for b in app.button]
    assert not any(label.startswith(("Execute plan", "Reject plan")) for label in labels)
    assert app.main.caption[-1].value == "Nothing was executed. Write a new request to continue."


def test_a_request_too_long_is_shown_before_the_plan(app_env: Settings) -> None:
    app = open_page("analysis.py")
    app.text_area(key="request_text").input(" ".join(["word"] * (MAX_REQUEST_WORDS + 1)))
    button(app.run(), "Generate plan").click().run()
    assert any("blocked" in m.value for m in app.markdown if "Run `" in m.value)
    # R01 stops the request before the planner: its card is in its own section, with
    # no Plan (no planner error either), no Validation and nothing to decide.
    assert [h.value for h in app.subheader] == ["Request check"]
    (card,) = [m.value for m in app.markdown if "REQUEST_TOO_LONG" in m.value]
    assert card.endswith("**R01 REQUEST_TOO_LONG** · request")
    assert not app.error
    labels = [b.label for b in app.button]
    assert not any(label.startswith(("Execute plan", "Reject plan")) for label in labels)
    assert app.main.caption[-1].value == (
        "The planner was not called and nothing was executed. Write a new request to continue."
    )


def test_scenario_4_is_rejected_whole_with_a_banner_and_no_controls(app_env: Settings) -> None:
    scenario = SCENARIOS_BY_ID["prompt_injection_unknown_capability"]
    app = open_page("analysis.py")
    assert app.sidebar.button(key=f"scenario-{scenario.id}").label == "4 · Prompt injection"
    app.sidebar.button(key=f"scenario-{scenario.id}").click().run()
    button(app, "Generate plan").click().run()
    # The planner output is labelled as simulated, above the plan.
    assert scenario.note.startswith("Simulated misbehaving planner")
    assert [w.value for w in app.warning] == [scenario.note]
    # Rejected whole by V02. The Plan section only says which step is blocked (as
    # escaped for Markdown, ui.escape_md): the banner and the step name no guardrail.
    (banner,) = [e.value for e in app.error]
    assert banner == (
        "Plan rejected: Step 1 · Run python is blocked\\. Nothing was executed; Validation "
        "below says why. Please write a new request."
    )
    (step,) = app.expander
    assert step.label.startswith("Step 1 · Run python") and "Blocked" in step.label
    assert "Blocked: it can be neither run nor edited (see Validation)." in [
        c.value for c in step.caption]  # fmt: skip
    # V02 is shown under Validation, alone, and nothing can be edited, approved,
    # executed or rejected: no field, approval box or button for it.
    assert "Validation" in [h.value for h in app.subheader]
    issues = [m.value for m in app.markdown if "UNKNOWN_CAPABILITY" in m.value]
    assert len(issues) == 1 and "V02" in issues[0]
    assert not app.number_input and not app.checkbox
    assert not [s for s in app.selectbox if s.key and s.key.startswith("param-")]
    labels = [b.label for b in app.button]
    for hidden in ("Apply changes", "Reset to planner", "Execute plan", "Reject plan"):
        assert not any(label.startswith(hidden) for label in labels), hidden
    assert "Nothing was executed. Write a new request to continue." in [
        c.value for c in app.caption]  # fmt: skip
    assert "Results" not in [h.value for h in app.subheader]
    assert "steps" not in app.session_state["run"].state  # never executed
    # A canned plan without a note (scenario 3's refusal) shows no such notice.
    app.sidebar.button(key="scenario-arima_forecast_block").click().run()
    button(app, "Generate plan").click().run()
    assert not app.warning


def test_execute_is_green_and_reject_red(app_env: Settings) -> None:
    app = open_page("analysis.py")
    app.sidebar.button(key="scenario-buy_the_dip_tech").click().run()
    button(app, "Generate plan").click().run()
    # Each decision button has a key, whose CSS class the page's style block colours.
    assert (app.button(key="execute").label, app.button(key="reject-plan").label) == (
        "Execute plan", "Reject plan")  # fmt: skip
    (style,) = [h.proto.body for h in app.get("html") if "st-key-execute" in h.proto.body]
    assert ".st-key-execute button:not(:disabled)" in style and "#188038" in style  # green
    assert ".st-key-reject-plan button:not(:disabled)" in style and "#d93025" in style  # red


def test_the_reviewer_can_reject_a_plan(app_env: Settings) -> None:
    app = open_page("analysis.py")
    app.sidebar.button(key="scenario-buy_the_dip_tech").click().run()
    button(app, "Generate plan").click().run()
    button(app, "Reject plan").click().run()
    assert not app.exception, app.exception
    assert button(app, "Execute plan").disabled and button(app, "Reject plan").disabled
    assert any(i.value.startswith("You rejected this plan") for i in app.info)
    [record] = load_runs(app_env.runs_dir)
    assert (record.status, record.decision, record.interactive) == ("rejected", "rejected", True)


@pytest.mark.parametrize(
    ("scenario_id", "enforcer", "category"),
    [
        ("buy_the_dip_tech", "Validator · Type: Deterministic", "Methodology"),
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
    scenario = SCENARIOS_BY_ID["buy_the_dip_tech"]
    invented = "Volatility peaked at 123.456 during the flagged weeks."
    # Live mode with a fake LLM: the canned plan, then the narrative that invents a number.
    plan = tool_response(SCENARIOS_BY_ID["buy_the_dip_tech"].plan)
    client = FakeAnthropicClient([plan, text_response(invented)])
    graph = build_graph()
    run = start_run(graph, scenario.request, list(scenario.dataset_ids), app_env, "live", True,
                    client)  # fmt: skip
    run = resume_run(graph, run, {"action": "approve", "issue_ids": SCENARIO_5_APPROVALS})

    app = AppTest.from_file(str(PAGES / "analysis.py"), default_timeout=60)
    app.session_state["run"], app.session_state["graph"] = run, graph
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
        "Name", "Action", "Fixable by edit", "Enforced by", "Type", "Category",
        "Description"]  # fmt: skip
    # Read from the guardrail definitions: V01 and V02 reject the plan.
    assert table["Fixable by edit"].tolist() == [g.fixable_text for g in GUARDRAILS.values()]
    assert (table.loc["V02", "Fixable by edit"], table.loc["V08", "Fixable by edit"]) == (
        "no", "yes")  # fmt: skip
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


def test_category_filter_explains_and_selects_each_category(app_env: Settings) -> None:
    app = open_page("guardrails.py")
    category = app.multiselect(key="guardrail_category")
    for name, definition in CATEGORY_DEFINITIONS.items():  # the tooltip, from the one list
        assert f"**{name}**: {definition}" in category.help
    category.select("Input validity").run()
    # The request (R01) and the data file (E01) are inputs too.
    assert guardrail_codes(app) == ["R01", "V03", "V04", "E01"]
    assert list(app.table[0].value["Category"].unique()) == ["Input validity"]


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


def plan_scenario_5(app_env: Settings) -> AppTest:
    """The analysis page with scenario 5's plan generated (buy the dip)."""
    app = open_page("analysis.py")
    app.sidebar.button(key="scenario-buy_the_dip_tech").click().run()
    button(app, "Generate plan").click().run()
    return app


def test_weekly_to_daily_edit_re_validates_and_runs_after_re_approval(app_env: Settings) -> None:
    app = plan_scenario_5(app_env)
    assert any("adjust parameters below" in i.value for i in app.info)
    history = next(n for n in app.number_input if n.key.endswith("-s4-min_periods"))
    # The label is the parameter's name; what its number is counted in is in the tooltip.
    assert history.label == "Minimum history" and history.value == 52
    assert "counted in periods of the input's frequency" in history.help

    next(s for s in app.selectbox if s.key.endswith("-s2-freq")).select("D").run()
    assert button(app, "Execute plan").disabled  # the change is not applied yet
    assert any(c.value == "Apply or reset your parameter changes to continue." for c in app.caption)

    button(app, "Apply changes").click().run()
    assert issue_codes(app) == {"V10"}  # the return-method approvals again, nothing else
    assert any("Approvals were reset" in i.value for i in app.info)
    assert any("edited (was Weekly)" in c.value for c in app.caption)

    approve_all(app)
    button(app, "Execute plan").click().run()
    assert not app.exception, app.exception
    trace = next(d.value for d in app.dataframe if "library call" in d.value.columns)
    params = dict(zip(trace["step"], trace["params"], strict=True))
    assert "Frequency: Daily (edited, was Weekly)" in params["s2"]
    assert "(edited" not in params["s4"]  # planner-proposed values are not marked
    # Under the trace, a grey note says what the minimum history is counted in.
    assert any(c.value == f"Minimum history: m{PERIODS_NOTE[1:]}" for c in app.caption)


def test_one_click_applies_a_typed_value(app_env: Settings) -> None:
    app = plan_scenario_5(app_env)
    # Apply is enabled before any change is seen: a number field sends a typed value only
    # with the next interaction, so a disabled button would swallow the first click.
    assert not button(app, "Apply changes").disabled
    button(app, "Apply changes").click().run()  # nothing changed yet
    assert any(i.value.startswith("No changes to apply") for i in app.info)
    assert issue_codes(app) == {"V10"}
    # The typed value and the click arrive in the same run: one click applies it.
    history = next(n for n in app.number_input if n.key.endswith("-s4-min_periods"))
    history.set_value(104)
    button(app, "Apply changes").click().run()
    assert any("Changes applied and re-validated" in i.value for i in app.info)
    assert any(c.value == ":blue[edited (was 52)]" for c in app.caption)


def test_apply_is_highlighted_only_while_there_are_unapplied_edits(app_env: Settings) -> None:
    app = plan_scenario_5(app_env)

    def apply_style() -> str:
        return app.button(key="apply-edits").proto.type

    def quantile():  # the field of the plan version shown now
        return next(n for n in app.number_input if n.key.endswith("-s4-quantile"))

    assert apply_style() == "secondary"  # no edits: white
    quantile().set_value(0.05).run()
    assert apply_style() == "primary"  # an edit not applied yet: red
    app.button(key="apply-edits").click().run()
    assert any("Changes applied and re-validated" in i.value for i in app.info)
    assert apply_style() == "secondary"  # the applied plan is the new baseline
    quantile().set_value(0.1).run()
    assert apply_style() == "primary"
    quantile().set_value(0.05).run()  # set back to the applied value: nothing to apply
    assert apply_style() == "secondary"
    assert not any("not applied yet" in c.value for c in app.caption)


def test_execute_ignores_a_click_while_a_change_is_not_applied(app_env: Settings) -> None:
    app = plan_scenario_5(app_env)
    approve_all(app)
    # A value typed, then Execute clicked straight away: both reach the same run, where
    # the unapplied change disables Execute, so Streamlit does not count the click.
    next(n for n in app.number_input if n.key.endswith("-s4-min_periods")).set_value(104)
    button(app, "Execute plan").click().run()
    assert not app.exception, app.exception
    assert "Results" not in [h.value for h in app.subheader]  # nothing ran
    assert any(c.value == "Apply or reset your parameter changes to continue." for c in app.caption)


def test_full_sample_edit_raises_v09_and_reset_clears_it(app_env: Settings) -> None:
    app = plan_scenario_5(app_env)
    next(s for s in app.selectbox if s.key.endswith("-s4-method")).select("full_sample").run()
    button(app, "Apply changes").click().run()
    assert issue_codes(app) == {"V09", "V10"}
    button(app, "Reset to planner").click().run()
    assert issue_codes(app) == {"V10"}


def step_headers(app: AppTest) -> list[tuple[str, bool]]:
    """(label, open) of every plan-step expander, in plan order."""
    return [(e.label, e.proto.expanded) for e in app.expander if e.label.startswith("Step ")]


def test_plan_steps_are_collapsed_and_approvals_stay_in_validation(app_env: Settings) -> None:
    app = plan_scenario_5(app_env)
    headers = step_headers(app)
    assert [is_open for _, is_open in headers] == [False] * 6  # the V10 steps too
    assert headers[0][0] == "Step 1 · C01 · Load series · NASDAQCOM \\- Synthetic equity index"
    assert (
        headers[2][0] == "Step 3 · C03 · Compute returns · Return type: Simple, periods: 1"
    )  # no badge; periods is the registry's default, shown filled in
    assert headers[3][0] == (
        "Step 4 · C07 · Flag regime · Quantile: 0\\.1, Threshold from: Past data only, Flag: Below "
        "the threshold"
    )
    assert headers[4][0] == "Step 5 · C08 · Forward return · Return type: Simple"
    assert headers[5][0] == "Step 6 · C10 · HAC mean difference"  # no parameters: no summary
    # The V10 approvals are still required, in the Validation section, naming their steps.
    v10 = [m.value for m in app.markdown if "**V10 MATERIAL_ASSUMPTION**" in m.value]
    assert v10[0].endswith("· Step 3 · C03 · Compute returns (s3)")
    assert v10[1].endswith("· Step 5 · C08 · Forward return (s5)")
    assert len([c for c in app.checkbox if c.key.startswith("approve-")]) == 2
    assert button(app, "Execute plan").disabled
    # Collapsing is visual only: the editor still runs.
    assert next(n for n in app.number_input if n.key.endswith("-s4-min_periods")).value == 52


def test_edits_are_marked_and_period_counts_follow_the_frequency(app_env: Settings) -> None:
    app = plan_scenario_5(app_env)
    next(s for s in app.selectbox if s.key.endswith("-s2-freq")).select("D").run()
    button(app, "Apply changes").click().run()
    headers = step_headers(app)
    assert headers[1] == (
        "Step 2 · C02 · Resample · Frequency: Daily, How: Last :blue-badge[:material/edit: edited]",
        False,
    )
    # min_periods 52 now counts trading days: its tooltip says it is counted in periods
    # of the series frequency (the summary of a window shows the unit: see
    # test_step_summary_falls_back_to_every_parameter), and nothing warns.
    history = next(n for n in app.number_input if n.key.endswith("-s4-min_periods"))
    assert history.label == "Minimum history"
    assert "counted in periods of the input's frequency" in history.help
    assert history.value == 52
    assert not any("Warning]" in label for label, _ in headers)
    # The new V10 approvals of this plan version open nothing and add no badge.
    assert headers[2] == ("Step 3 · C03 · Compute returns · Return type: Simple, periods: 1", False)


def test_step_summary_falls_back_to_every_parameter(app_env: Settings) -> None:
    import ui  # importable once app_env has put app/ on the path

    catalog = load_catalog(app_env.catalog_path)
    flag = Step(id="s5", capability="flag_regime", inputs={"series": "s4"},
                params={"quantile": 0.9, "method": "expanding", "min_periods": 52})  # fmt: skip
    spec = REGISTRY["flag_regime"]
    assert (
        ui.step_summary(flag, spec, "W", catalog) == "Quantile: 0.9, Threshold from: Past data only"
    )
    every = dataclasses.replace(spec, summary_params=())
    assert ui.step_summary(flag, every, "W", catalog).endswith("Minimum history: 52 weeks")
    # A window counted in periods shows its unit, which follows the input's frequency.
    volatility = Step(**volatility_plan()["steps"][3])  # rolling_volatility, window 12
    summary = ui.step_summary(volatility, REGISTRY["rolling_volatility"], "D", catalog)
    assert summary == "Window: 12 trading days, Annualised: yes"
    unknown = Step(id="s2", capability="arima", params={"steps_ahead": 1})
    # An unregistered parameter has no label: its raw name is shown.
    assert ui.step_summary(unknown, None, None, catalog) == "steps_ahead: 1"
    header = ui.step_header(2, unknown, "steps ahead: 1", {Action.BLOCK}, edited=False)
    assert header == "Step 2 · Arima · steps ahead: 1 :red-badge[:material/block: Blocked]"


def test_unapplied_edits_compares_the_fields_with_the_last_validated_plan(
    app_env: Settings,
) -> None:
    import ui  # importable once app_env has put app/ on the path

    plan = as_plan(volatility_plan())
    fields = {step.id: dict(step.params) for step in plan.steps}  # every field as planned
    assert ui.unapplied_edits(plan, fields) == {}  # no change
    fields["s5"]["quantile"] = 0.95
    assert ui.unapplied_edits(plan, fields) == {"s5": {"quantile": 0.95}}  # a real change
    fields["s5"]["quantile"] = 0.9
    assert ui.unapplied_edits(plan, fields) == {}  # a change set back to the plan's value

    # Types count, as for the validator: a planner's 12.0 for the int window (V03)
    # differs from the field's 12, so applying the field's value fixes it.
    raw = volatility_plan()
    next(s for s in raw["steps"] if s["id"] == "s4")["params"]["window"] = 12.0
    assert ui.unapplied_edits(as_plan(raw), fields) == {"s4": {"window": 12}}
    assert not ui.same_value(True, 1)
    assert ui.same_value({"a": [0.9], "b": 1}, {"b": 1, "a": [0.9]})  # key order is no change


def test_flagged_spans_are_shaded_one_band_each(app_env: Settings) -> None:
    import ui  # importable once app_env has put app/ on the path

    days = pd.bdate_range("2024-01-01", periods=6)
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=days)
    flags = pd.Series([True, True, False, True, False, False], index=days)
    bands = ui.series_figure(values, "title", "name", flags).layout.shapes
    # Two flagged spans, each from its first date to the next observation's date, the
    # full height of the chart, behind the line.
    assert [(b.x0, b.x1) for b in bands] == [(days[0], days[2]), (days[3], days[4])]
    assert {(b.yref, b.y0, b.y1, b.layer) for b in bands} == {("y domain", 0, 1, "below")}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(SCENARIOS_BY_ID["chf_safe_haven_success"].plan, {"s3": "s2"}, id="scenario 1"),
        pytest.param(
            SCENARIOS_BY_ID["spread_frequency_mismatch_block"].plan, {"s5": "s4"}, id="scenario 2"
        ),  # fmt: skip
        pytest.param(SCENARIOS_BY_ID["buy_the_dip_tech"].plan, {"s4": "s3"}, id="scenario 5"),
        pytest.param(volatility_plan(), {"s5": "s4"}, id="volatility plan"),
    ],
)
def test_each_regime_chart_belongs_to_its_flag_regime_step(
    app_env: Settings, raw: dict, expected: dict[str, str]
) -> None:
    import ui  # importable once app_env has put app/ on the path

    # Found by capability, wherever the step is: flag_regime step -> its series input.
    plan = as_plan(raw)
    ran = {step.id: None for step in plan.steps}  # every step produced an output
    assert ui.regime_charts(plan.steps, ran) == expected
    # No chart without both the flags and their series.
    (flag_step,) = expected
    assert ui.regime_charts(plan.steps, {k: v for k, v in ran.items() if k != flag_step}) == {}
    no_series = {k: v for k, v in ran.items() if k != expected[flag_step]}
    assert ui.regime_charts(plan.steps, no_series) == {}


def test_the_regime_chart_is_in_the_open_flag_regime_section(app_env: Settings) -> None:
    app = plan_scenario_1(app_env)
    approve_all(app)
    button(app, "Execute plan").click().run()
    assert not app.exception, app.exception
    sections = {e.label.split(" · completed")[0]: e for e in app.expander if " · C" in e.label}
    flags, series = sections["s3 · C07 · Flag regime"], sections["s2 · C02 · Resample"]
    # The flag_regime section opens and holds the regime chart: the weekly VIX (its own
    # title) with the flagged weeks shaded.
    assert flags.proto.expanded
    (chart,) = flags.get("plotly_chart")
    spec = json.loads(chart.proto.spec)
    assert spec["layout"]["title"]["text"].startswith("s2 · C02 · Resample")
    assert spec["layout"]["shapes"]  # the flagged weeks
    # The series' own section stays collapsed and its chart is not shaded.
    assert not series.proto.expanded
    (plain,) = series.get("plotly_chart")
    assert not json.loads(plain.proto.spec)["layout"].get("shapes")


def test_every_registered_choice_has_a_label_that_round_trips(app_env: Settings) -> None:
    import labels  # importable once app_env has put app/ on the path

    resample = REGISTRY["resample"].param("freq")
    assert [labels.param_value_label("resample", "freq", c) for c in resample.choices] == [
        "Daily", "Weekly", "Monthly",
    ]  # fmt: skip
    assert labels.param_label("resample", "freq") == "Frequency"
    # resample's `how`: its name and its values are capitalised.
    how = REGISTRY["resample"].param("how")
    assert [labels.param_value_label("resample", "how", c) for c in how.choices] == [
        "Last", "Mean",
    ]  # fmt: skip
    assert labels.param_label("resample", "how") == "How"
    # Every labelled parameter: each allowed code has its own label, and the label leads
    # back to the code (label -> code -> label), so the editor can show labels and
    # still write codes.
    for spec in REGISTRY.values():
        for spec_param in spec.params:
            mapped = labels.VALUE_LABELS.get(
                (spec.name, spec_param.name)
            ) or labels.VALUE_LABELS.get((None, spec_param.name))
            if not mapped or not spec_param.choices:
                continue
            shown = {
                c: labels.param_value_label(spec.name, spec_param.name, c)
                for c in spec_param.choices
            }
            assert all(label != code for code, label in shown.items()), (spec.name, spec_param.name)
            to_code = {label: code for code, label in shown.items()}
            assert len(to_code) == len(shown)  # no two codes share a label
            for code, label in shown.items():
                assert to_code[label] == code  # label -> code
                assert labels.param_value_label(spec.name, spec_param.name, code) == label
    # Every parameter counted in periods: a short label, and a description (the editor's
    # tooltip) that says what its number is counted in; read-only views add the note.
    for spec in REGISTRY.values():
        for spec_param in spec.params:
            if spec_param.counted_in_periods:
                assert "(in periods" not in labels.param_label(spec.name, spec_param.name)
                assert "counted in periods of the input's frequency" in spec_param.description
                assert labels.param_note(spec.name, spec_param.name) == PERIODS_NOTE
    assert labels.param_note("flag_regime", "quantile") is None
    # Anything unmapped is shown as it is.
    assert labels.param_label("compute_returns", "periods") == "periods"
    assert labels.param_value_label("compute_returns", "periods", 1) == "1"
    assert labels.param_label("arima", "steps_ahead") == "steps_ahead"
    assert labels.stat_label("n_eff") == "n_eff" and labels.kind_label("new_kind") == "new_kind"


def test_the_editor_shows_labels_but_the_plan_keeps_its_codes(app_env: Settings) -> None:
    app = plan_scenario_5(app_env)
    headers = step_headers(app)
    assert headers[1][0] == "Step 2 · C02 · Resample · Frequency: Weekly, How: Last"
    freq = next(s for s in app.selectbox if s.key.endswith("-s2-freq"))
    assert freq.label == "Frequency" and freq.options == ["Daily", "Weekly", "Monthly"]
    assert freq.value == "W"  # the field's value is the code
    # Choosing the label already shown is no edit: Apply stays white, the plan unchanged.
    freq.select("W").run()
    assert app.button(key="apply-edits").proto.type == "secondary"
    # The same for `how`: the field shows "Last" and holds "last"; choosing "Last" again
    # marks nothing as edited or pending.
    how = next(s for s in app.selectbox if s.key.endswith("-s2-how"))
    assert how.label == "How" and how.options == ["Last", "Mean"] and how.value == "last"
    how.select_index(how.options.index("Last")).run()
    assert app.button(key="apply-edits").proto.type == "secondary"
    assert not any("not applied yet" in c.value for c in app.caption)
    assert step_headers(app)[1][0] == "Step 2 · C02 · Resample · Frequency: Weekly, How: Last"
    assert app.session_state["run"].state["plan"].steps[1].params["how"] == "last"
    # A real edit stores the code.
    next(s for s in app.selectbox if s.key.endswith("-s2-freq")).select("D").run()
    assert app.button(key="apply-edits").proto.type == "primary"
    app.button(key="apply-edits").click().run()
    assert app.session_state["run"].state["plan"].steps[1].params["freq"] == "D"


def chart_bars(app: AppTest) -> list[int]:
    """The number of bars (or points) in each plotly chart of the page, in order."""
    return [
        sum(len(trace.get("y", [])) for trace in json.loads(chart.proto.spec)["data"])
        for chart in app.get("plotly_chart")
    ]


def tinted_cards(node, status: str) -> list[str]:
    """Labels of the KPI cards inside a container keyed "kpi-<status>-..." (tinted by CSS)."""
    labels = []
    for child in getattr(node, "children", {}).values():
        if f"-kpi-{status}-" in getattr(child.proto, "id", ""):
            labels += [metric.proto.label for metric in child.get("metric")]
        else:
            labels += tinted_cards(child, status)
    return labels


def test_monitoring_page_with_no_runs_shows_a_note(app_env: Settings) -> None:
    app = open_page("monitoring.py")
    assert app.info[0].value.startswith("No runs yet")
    assert not app.get("plotly_chart") and not app.metric and not app.subheader


def headless_replayed_run(settings: Settings) -> str:
    """A headless run (as the CLI makes), replayed once; returns its run id."""
    scenario = SCENARIOS_BY_ID["buy_the_dip_tech"]
    state = run_graph(scenario.request, list(scenario.dataset_ids), "mock", approve=True,
                      settings=settings)  # fmt: skip
    assert replay(state["run_id"], settings).all_match
    return state["run_id"]


def test_monitoring_page_sections_cards_and_charts(app_env: Settings) -> None:
    # Counted: approved and executed, rejected, blocked (V08, with a V09 to approve), refused
    # by the planner (P01), failed, and a live run that completed with a warning (E04).
    # Runs started on the Analysis page (a person reviews): each pauses for review.
    graph, run = start_review("buy_the_dip_tech", app_env)
    execute(graph, run, paused_ids(run))
    graph, run = start_review("buy_the_dip_tech", app_env)
    resume_run(graph, run, {"action": "reject"})
    start_review("spread_frequency_mismatch_block", app_env)
    start_review("arima_forecast_block", app_env)
    # Left out: a plan still awaiting a decision, and a headless run (whose replay counts).
    start_review("buy_the_dip_tech", app_env)
    headless_replayed_run(app_env)
    now = utc_now()
    save_run(RunRecord(run_id=new_run_id(), created_at=now, status="failed", request="x",
                       error="boom", stage_timings_ms={"planning": 5.0}, interactive=True,
                       llm=LLMInfo(provider="mock", model="mock", prompt_version="canned",
                                   mode="mock")),
             {}, app_env.runs_dir)  # fmt: skip
    luna = {"model": "gpt-6-luna"}
    save_run(RunRecord(run_id=new_run_id() + "x", created_at=now, status="completed",
                       request="y", llm=LLMInfo(provider="openai", prompt_version="planner-v6",
                                                mode="live", **luna),
                       warnings=[make_issue("E04", "s1", "missing values", key="values")],
                       stage_timings_ms={"planning": 1200.0, "narration": 800.0},
                       stage_tokens={"planning": TokenUsage(input_tokens=1500, output_tokens=400,
                                                            **luna),
                                     "narration": TokenUsage(input_tokens=1000,
                                                             output_tokens=200, **luna)},
                       interactive=True),
             {}, app_env.runs_dir)  # fmt: skip

    app = open_page("monitoring.py")
    assert [h.value for h in app.subheader] == ["Activity", "Quality and governance",
        "Capabilities", "Guardrails", "Cost", "Latency", "Recent errors"]  # fmt: skip
    assert not app.info  # there are counted runs
    cards = {m.proto.label: m.proto for m in app.metric}
    outcomes = ("Completed", "Blocked", "Planner refused (P01)", "Rejected by human", "Failed")
    values = {label: cards[label].body for label in ("Total runs", *outcomes)}
    # The undecided and headless runs are left out. The five outcomes do not overlap, and
    # their whole percentages add up to 100% (rounded on their own, 33 + 4 x 17 = 101).
    assert values == {"Total runs": "6", "Completed": "2 (33%)", "Blocked": "1 (17%)",
                      "Planner refused (P01)": "1 (17%)", "Failed": "1 (17%)",
                      "Rejected by human": "1 (16%)"}  # fmt: skip
    assert sum(int(re.search(r"\((\d+)%\)", values[label])[1]) for label in outcomes) == 100
    # The activity cards, then the quality and governance cards, in order: six and six.
    assert [m.proto.label for m in app.metric][:12] == [
        "Total runs", *outcomes,
        "Approval rate", "Edit rate", "Approval time: median, p95", "Completed with warnings",
        "Grounded narrative (N01)", "Replay match rate"]  # fmt: skip
    # Every card says what its number is out of; Total runs splits by LLM mode.
    assert cards["Total runs"].delta == "1 live, 0 cached, 5 mock"
    assert all(cards[label].delta == "of 6 runs" for label in outcomes)
    # Each quality card: the absolute number, then the percentage in brackets.
    # One approved, one rejected: approved / (approved + rejected).
    assert (cards["Approval rate"].body, cards["Approval rate"].delta) == (
        "1 (50%)", "of 2 plans reviewed")  # fmt: skip
    assert (cards["Edit rate"].body, cards["Edit rate"].delta) == ("0 (0%)", "of 1 plan approved")
    # The median, then the 95th percentile.
    time = cards["Approval time: median, p95"]
    assert re.fullmatch(r"\d+\.\d s, \d+\.\d s", time.body)
    assert time.delta == "over 1 plan approved"
    assert (cards["Grounded narrative (N01)"].body, cards["Grounded narrative (N01)"].delta) == (
        "1 (100%)", "of 1 narrative")  # fmt: skip
    # Out of the completed runs: the approved run, and the live run with its E04 warning.
    warned = cards["Completed with warnings"]
    assert (warned.body, warned.delta) == ("1 (50%)", "of 2 completed runs")
    # The replay of the headless run counts: replays come from every source.
    assert (cards["Replay match rate"].body, cards["Replay match rate"].delta) == (
        "1 (100%)", "of 1 replay")  # fmt: skip
    # Cards with a target: orange when missed (its note orange, a warning icon on its
    # title), green when met.
    assert cards["Failed"].color == 3  # 3: ORANGE
    assert cards["Failed"].icon == ":material/warning:"
    assert tinted_cards(app.main, "missed") == ["Failed"]
    assert tinted_cards(app.main, "met") == ["Grounded narrative (N01)", "Replay match rate"]
    # Every capability; the guardrails that fired in the counted runs; cost per stage; mean
    # and p95 for each stage with data (all but the untimed request check).
    runs = monitored_runs(load_runs(app_env.runs_dir))
    events = monitored_events(read_events(app_env.events_path), runs)
    fired = guardrail_counts(events, include_unused=False)
    assert {r["code"] for r in fired} == {"V10", "V08", "V09", "P01"}  # not the undecided plan's
    assert chart_bars(app) == [len(REGISTRY), len(fired), 2, 2 * 4]
    # The unused guardrails: a compact table sorted by code, no chart.
    unused = next(e for e in app.expander if e.label == "Show unused guardrails")
    assert not unused.get("plotly_chart")
    used = {"V10", "V08", "V09", "P01"}
    assert unused.dataframe[0].value["Code"].tolist() == sorted(set(GUARDRAILS) - used)
    cost, latency = [t.value for t in app.table]
    assert cost.loc["planner", "Cost, US$ (mean)"] == pytest.approx(0.00035)
    assert cost.loc["total", "Input tokens (mean)"] == 2500
    assert latency.index.tolist() == [
        "Request check", "Planner", "Validator", "Executor", "Narrator"]  # fmt: skip
    assert pd.isna(latency.loc["Request check", "Mean (ms)"])  # not timed: shown as a dash
    # Dollar signs in a caption are escaped, or a pair of them would render as LaTeX.
    (prices,) = [c.value for c in app.caption if c.value.startswith("Per run")]
    assert r"Total \$0.000550" in prices and r"gpt-6-luna \$0.10 input" in prices
    errors = next(d.value for d in app.dataframe if "stage" in d.value.columns)
    assert errors["code"].tolist() == ["—", "P01", "V08"] and errors["message"][0] == "boom"


def test_monitoring_page_with_no_counted_run_still_shows_replays(app_env: Settings) -> None:
    headless_replayed_run(app_env)
    start_review("buy_the_dip_tech", app_env)  # awaiting a decision
    app = open_page("monitoring.py")
    assert app.info[0].value.startswith("No decided runs from the Analysis page yet")
    cards = {m.proto.label: m.proto for m in app.metric}
    assert cards["Total runs"].body == "0" and cards["Completed"].body == "0 (n/a)"
    assert cards["Approval time: median, p95"].body == "n/a"  # no p95 without data
    assert (cards["Replay match rate"].body, cards["Replay match rate"].delta) == (
        "1 (100%)", "of 1 replay")  # fmt: skip
    assert cards["Total runs"].delta == "0 live, 0 cached, 0 mock"
    # No counted run: the Failed card has no data, so it is not green.
    assert tinted_cards(app.main, "met") == ["Replay match rate"]
    assert tinted_cards(app.main, "missed") == []


def test_monitoring_page_before_any_guardrail_fired(app_env: Settings) -> None:
    save_run(RunRecord(run_id=new_run_id(), created_at=utc_now(), status="completed",
                       request="x", interactive=True), {}, app_env.runs_dir)  # fmt: skip
    app = open_page("monitoring.py")
    assert any(c.value == "No guardrail has fired yet." for c in app.caption)
    # The expander lists every guardrail, none having fired.
    unused = next(e for e in app.expander if e.label == "Show unused guardrails")
    assert len(unused.dataframe[0].value) == len(GUARDRAILS)


# ---------------------------------------------------------------- change requests

SPREAD = SCENARIOS_BY_ID["spread_frequency_mismatch_block"]


def live_page(
    app_env: Settings, responses: list, scenario_id: str = SPREAD.id, mode: str = "live"
) -> AppTest:
    """The Analysis page on a run of an example in `mode` (live by default), its LLM a fake
    serving the example's plan, then `responses`."""
    scenario = SCENARIOS_BY_ID[scenario_id]
    client = FakeAnthropicClient([tool_response(scenario.plan), *responses])
    graph = build_graph()
    run = start_run(graph, scenario.request, list(scenario.dataset_ids), app_env, mode, True,
                    client)  # fmt: skip
    app = AppTest.from_file(str(PAGES / "analysis.py"), default_timeout=60)
    app.session_state["llm_mode"] = mode
    app.session_state["run"], app.session_state["graph"] = run, graph
    app.run()
    assert not app.exception, app.exception
    return app


def request_changes(app: AppTest, comment: str) -> None:
    """Type a change request into the comment box and click Request changes."""
    state = app.session_state["run"].state
    revision = len(state["revisions"]) - 1
    app.text_area(key=f"comment-{state['run_id']}-{revision}").input(comment).run()
    button(app, "Request changes").click().run()
    assert not app.exception, app.exception


def grey_styles(app: AppTest) -> list[str]:
    """The style blocks that colour the Request changes button grey (ready to send)."""
    return [h.proto.body for h in app.get("html") if "#5f6368" in h.proto.body]


def test_a_change_request_revises_the_plan_on_the_page(app_env: Settings) -> None:
    narrative = text_response("The flagged weeks are the most inverted of the whole history.")
    app = live_page(app_env, [tool_response(spread_weekly_plan()), narrative])
    # Blocked by V08: no Execute, but Execute | Request changes | Reject are offered.
    assert button(app, "Execute plan").disabled and not button(app, "Request changes").disabled
    request_changes(app, CHANGE_REQUEST_COMMENT)
    # The revision: its number, the comment, and the diff of steps and parameters.
    badge = next(m.value for m in app.markdown if "Revision 1 of 3" in m.value)
    assert CHANGE_REQUEST_COMMENT in badge
    changes = next(e for e in app.expander if e.label == "Changes from revision 0")
    rows = changes.dataframe[0].value.to_dict("records")
    assert rows == [
        {"step": "s4", "change": "added", "details": "Resample (series <- s3; freq=W, how=last)"},
        {"step": "s4 -> s5", "change": "changed", "details": "input b: s3 -> s4"},
    ]
    # V08 is gone; the look-ahead approval (V09) remains, then Execute runs it.
    assert not any("FREQUENCY_MISMATCH" in m.value for m in app.markdown)
    approve_all(app)
    button(app, "Execute plan").click().run()
    assert app.session_state["run"].state["status"] == "completed"
    # The trace shows how the plan came about, and still traces the executed plan.
    assert any(m.value == "**Revision history**" for m in app.markdown)
    history = next(d.value for d in app.dataframe if "comment" in d.value.columns)
    assert history["comment"].tolist() == ["(the planner's first proposal)", CHANGE_REQUEST_COMMENT]
    assert history["validation"].tolist() == ["blocked: V08, V09", "ready: V09"]


@pytest.mark.parametrize("mode", ["mock", "cached"])
def test_request_changes_is_live_only_on_the_page(mode: str, app_env: Settings) -> None:
    if mode == "mock":
        app = open_page("analysis.py", llm_mode="mock")
        app.sidebar.button(key=f"scenario-{SPREAD.id}").click().run()
        button(app, "Generate plan").click().run()
    else:  # the cache is empty, so the fake LLM answers the first plan
        app = live_page(app_env, [], mode="cached")
    change = button(app, "Request changes")
    assert change.disabled and change.help == "Available in live mode."


def test_after_the_last_change_request_only_execute_and_reject_remain(app_env: Settings) -> None:
    revisions = [tool_response(SPREAD.plan) for _ in range(3)]  # still blocked each time
    app = live_page(app_env, revisions)
    for number in range(3):
        request_changes(app, f"try again, attempt {number + 1}")
    assert any("Revision 3 of 3" in m.value for m in app.markdown)
    labels = [b.label for b in app.button]
    assert "Request changes" not in labels and "Execute plan" in labels and "Reject plan" in labels
    assert not [t for t in app.text_area if t.key.startswith("comment-")]


def test_a_refusal_can_be_sent_back_with_a_change_request(app_env: Settings) -> None:
    revision = tool_response(spread_weekly_plan())
    app = live_page(app_env, [revision], scenario_id="arima_forecast_block")
    declined = "The planner declined this request"
    assert app.error[0].proto.title == declined
    assert not button(app, "Request changes").disabled
    assert app.main.caption[-1].value == (
        "Nothing was executed. Request changes above, or write a new request."
    )
    request_changes(app, CHANGE_REQUEST_COMMENT)
    # The planner's revision replaces the refusal: a plan to review and validate.
    assert any("Revision 1 of 3" in m.value for m in app.markdown)
    assert declined not in [e.proto.title for e in app.error]
    assert "Validation" in [h.value for h in app.subheader]
    # The new revision has a new button: a click queued on the old one while the planner
    # was revising cannot send the comment a second time.
    run_id = app.session_state["run"].state["run_id"]
    assert button(app, "Request changes").key == f"request-changes-{run_id}-1"


def test_streamlit_runs_one_decision_at_a_time() -> None:
    # A click made while the LLM answers waits, instead of resuming the same run twice.
    with open(PROJECT_ROOT / ".streamlit" / "config.toml", "rb") as config:
        assert tomllib.load(config)["runner"]["fastReruns"] is False


@pytest.mark.parametrize("scenario_id", [SPREAD.id, "arima_forecast_block"])
def test_request_changes_turns_grey_once_a_comment_is_written(
    scenario_id: str, app_env: Settings
) -> None:
    app = live_page(app_env, [], scenario_id=scenario_id)  # a plan, and a refusal (P01)
    comment = app.text_area(key=f"comment-{app.session_state['run'].state['run_id']}-0")
    assert grey_styles(app) == []  # no comment yet: the neutral button
    comment.input(CHANGE_REQUEST_COMMENT).run()
    (style,) = grey_styles(app)
    assert f".st-key-{button(app, 'Request changes').key} button:not(:disabled)" in style
    comment.input("   ").run()  # blank is no comment
    assert grey_styles(app) == []


def test_request_changes_is_not_grey_while_a_parameter_change_is_pending(
    app_env: Settings,
) -> None:
    app = live_page(app_env, [])
    comment = app.text_area(key=f"comment-{app.session_state['run'].state['run_id']}-0")
    comment.input(CHANGE_REQUEST_COMMENT).run()
    next(n for n in app.number_input if n.key.endswith("-quantile")).set_value(0.2).run()
    assert button(app, "Request changes").disabled and grey_styles(app) == []
