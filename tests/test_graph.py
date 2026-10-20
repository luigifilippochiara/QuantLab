"""Graph: the orchestrator, from request to saved record.

The request check, the planner, the validator, the approval interrupt (a run pauses
and resumes from its checkpoint with the same state), execution, narration and the
run record. Editing is tested in test_editing.py, change requests in
test_change_requests.py, the examples in test_mock_scenarios.py, and the examples'
results against the golden record in test_golden_scenarios.py.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from quantlab.config import Settings
from quantlab.graph import (
    build_graph,
    current_plan_hash,
    issue_codes,
    refusal_reason,
    resume_run,
    run_graph,
    start_run,
)
from quantlab.guardrails import MAX_REQUEST_CHARS, MAX_REQUEST_WORDS
from quantlab.lineage import RunRecord, load_run, load_runs, replay
from quantlab.mock_scenarios import SCENARIOS, SCENARIOS_BY_ID, Scenario
from quantlab.monitoring import activity, read_events
from quantlab.planner import PROMPT_VERSION
from quantlab.types import TokenUsage, artifact_hash

from conftest import SCENARIO_5_APPROVALS, edit, execute, paused_ids, raw_plan, start_review
from fakes import FakeAnthropicClient, text_response, tool_response

SUCCESS = SCENARIOS_BY_ID["buy_the_dip_tech"]  # scenario 5


# --- a run from request to record ------------------------------------------------


def test_end_to_end_scenario_5_with_manifest_and_replay(settings: Settings) -> None:
    graph, run = start_review(SUCCESS.id, settings)
    assert run.state["status"] == "ready" and paused_ids(run) == SCENARIO_5_APPROVALS
    # Execute without the approvals is refused, and the run asks again.
    refused = execute(graph, run, [])
    assert refused.paused_for["error"].startswith("Approval required")

    run = execute(graph, refused, SCENARIO_5_APPROVALS)
    state = run.state
    assert run.paused_for is None and state["status"] == "completed"
    narrative = state["narrative"]
    assert narrative.grounding.grounded
    manifest = Path(state["manifest_path"])
    record = load_run(state["run_id"], settings.runs_dir)
    assert manifest.exists() and record.status == "completed"
    assert record.narrative["note"] == narrative.note
    # Each issue in the manifest names the stage that enforced it and its type.
    saved = json.loads(manifest.read_text(encoding="utf-8"))["validation"]["issues"]
    assert {(i["code"], i["enforced_by"], i["type"]) for i in saved} == {
        ("V10", "Validator", "Deterministic")}  # fmt: skip
    assert [a.issue_id for a in record.approvals] == SCENARIO_5_APPROVALS
    assert record.llm.provider == "mock" and record.registry_version
    assert record.plan_original["steps"][3]["params"]["quantile"] == 0.10  # as proposed
    # The planner's plan is kept as it came (example 5's omits parameters with a
    # default); the plan that ran has the registry's defaults filled in, nothing else.
    original, final = record.plan_original["steps"][3], record.plan_final["steps"][3]
    assert "min_periods" not in original["params"] and final["params"]["min_periods"] == 52
    assert final["params"] == {**original["params"], "method": "expanding", "min_periods": 52}
    assert set(record.stage_timings_ms) == {"planning", "validation", "execution", "narration"}

    report = replay(state["run_id"], settings)
    assert report.all_match, report.message


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
def test_a_headless_run_records_one_revision_and_no_change_request(
    scenario: Scenario, settings: Settings
) -> None:
    approve = scenario.auto_approve
    state = run_graph(scenario.request, list(scenario.dataset_ids), "mock", approve, settings)
    record = load_run(state["run_id"], settings.runs_dir)
    assert record == state["record"]  # the record on disk is the graph's final record
    # The planner was asked once; nobody changed the plan.
    (revision,) = record.revisions
    assert revision.comment is None and revision.validation == record.validation
    assert revision.plan_hash == record.plan_hash
    assert (record.change_requests, record.manual_edits) == (0, 0)


def test_a_headless_run_without_approval_executes_nothing(settings: Settings) -> None:
    state = run_graph(SUCCESS.request, list(SUCCESS.dataset_ids), "mock", settings=settings)
    assert state["status"] == "blocked" and "steps" not in state
    assert "Approval required" in load_run(state["run_id"], settings.runs_dir).error


def test_executing_is_the_approval_decision(settings: Settings) -> None:
    state = run_graph(SUCCESS.request, list(SUCCESS.dataset_ids), "mock", True, settings)
    record = load_run(state["run_id"], settings.runs_dir)
    assert record.decision == "approved" and record.interactive is False  # a headless run
    assert record.plan_shown_at <= record.decided_at
    assert {a.approved_by for a in record.approvals} == {"cli"}
    assert record.stage_tokens == {"planning": TokenUsage(), "narration": TokenUsage()}


def test_a_blocked_scenario_is_recorded(settings: Settings) -> None:
    scenario = SCENARIOS_BY_ID["arima_forecast_block"]
    state = run_graph(scenario.request, list(scenario.dataset_ids), "mock", settings=settings)
    assert state["status"] == "blocked"
    record = load_run(state["run_id"], settings.runs_dir)
    assert record.status == "blocked" and "P01" in record.error
    events = read_events(settings.events_path)
    assert events[-1]["event_type"] == "run_failed"
    assert events[-1]["details"]["status"] == "blocked"


def test_monitoring_sees_every_stage(settings: Settings) -> None:
    run_graph(SUCCESS.request, list(SUCCESS.dataset_ids), "mock", True, settings)
    injection = SCENARIOS_BY_ID["prompt_injection_unknown_capability"]
    run_graph(injection.request, list(injection.dataset_ids), "mock", settings=settings)
    events = read_events(settings.events_path)
    types = {e["event_type"] for e in events}
    assert {"request_received", "plan_generated", "plan_validated", "approval_granted",
            "execution_started", "step_completed", "execution_finished",
            "narrative_checked", "run_completed", "run_failed"} <= types  # fmt: skip
    summary = activity(load_runs(settings.runs_dir))
    assert (summary["total"], summary["completed"], summary["blocked"]) == (2, 1, 1)


# --- the approval interrupt and the checkpoint ----------------------------------


def test_a_run_pauses_at_approval_and_resumes_with_the_same_state(settings: Settings) -> None:
    graph = build_graph()
    run = start_run(graph, SUCCESS.request, list(SUCCESS.dataset_ids), settings)
    snapshot = graph.get_state(run.config)
    assert snapshot.next == ("approval",) and paused_ids(run) == SCENARIO_5_APPROVALS
    # The checkpoint gives back the very same state, rebuilt from its stored form.
    assert snapshot.values == run.state and run.state["status"] == "ready"
    # A run awaiting a decision is on record.
    record = load_run(run.state["run_id"], settings.runs_dir)
    assert record.status == "ready" and record.decision is None and record.plan_shown_at

    paused_hash = current_plan_hash(snapshot.values)
    run = resume_run(graph, run, {"action": "approve", "issue_ids": SCENARIO_5_APPROVALS})
    assert run.paused_for is None and run.state["status"] == "completed"
    assert current_plan_hash(run.state) == paused_hash
    assert {a.plan_hash for a in run.state["approvals"]} == {paused_hash}
    assert run.state["narrative"].grounding.grounded
    assert replay(run.state["run_id"], settings).all_match


def test_the_checkpoint_round_trips_the_whole_state(settings: Settings) -> None:
    graph = build_graph()
    state = run_graph(SUCCESS.request, list(SUCCESS.dataset_ids), "mock", True, settings,
                      graph=graph)  # fmt: skip
    saved = graph.get_state({"configurable": {"thread_id": state["run_id"]}}).values
    assert isinstance(saved["record"], RunRecord)
    # Artifacts hold pandas Series, so they are compared by hash and metadata.
    assert {k: v for k, v in saved.items() if k != "artifacts"} == {
        k: v for k, v in state.items() if k != "artifacts"
    }
    for step_id, artifact in state["artifacts"].items():
        assert artifact_hash(saved["artifacts"][step_id]) == artifact_hash(artifact)
        assert saved["artifacts"][step_id].meta == artifact.meta


def test_every_checkpointed_class_is_allowlisted(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    # A class missing from CHECKPOINT_TYPES is logged when a run resumes, and comes
    # back as a plain dict (or, for an enum, a plain string).
    caplog.set_level(logging.WARNING)
    for scenario in SCENARIOS:
        run_graph(scenario.request, list(scenario.dataset_ids), "mock", True, settings)
    assert not [r.getMessage() for r in caplog.records if r.name.startswith("langgraph")]


def test_a_plan_shown_is_saved_while_it_awaits_a_decision(settings: Settings) -> None:
    graph, run = start_review(SUCCESS.id, settings)
    record = load_run(run.state["run_id"], settings.runs_dir)  # saved before any decision
    assert record.status == "ready" and record.interactive is True
    assert record.plan_shown_at and record.decision is None and record.decided_at is None
    assert record.stage_tokens == {"planning": TokenUsage()}  # mock: zero tokens
    # An edit re-validates and saves again, so the record always holds the current plan.
    edit(graph, run, {"s4": {"min_periods": 104}})
    assert load_run(run.state["run_id"], settings.runs_dir).param_diff[0].final == 104


def test_an_edit_goes_back_to_the_validator_and_asks_again(settings: Settings) -> None:
    graph = build_graph()
    run = start_run(graph, SUCCESS.request, list(SUCCESS.dataset_ids), settings)
    edits = {"s3": {"method": "log"}, "s4": {"min_periods": 104}}
    run = resume_run(graph, run, {"action": "edit", "edits": edits})
    # Re-validated: the same approvals are asked for again, now for the edited plan.
    assert graph.get_state(run.config).next == ("approval",)
    assert paused_ids(run) == SCENARIO_5_APPROVALS
    assert "method='log'" in run.paused_for["open_approvals"][0]["message"]
    assert [(e.step_id, e.param, e.old, e.new) for e in run.state["edits"]] == [
        ("s3", "method", "simple", "log"),
        ("s4", "min_periods", 52, 104),
    ]
    run = resume_run(graph, run, {"action": "approve", "issue_ids": SCENARIO_5_APPROVALS})
    assert run.state["status"] == "completed"
    assert run.state["artifacts"]["s3"].meta.return_method == "log"
    record = load_run(run.state["run_id"], settings.runs_dir)
    assert [(c.step_id, c.param, c.final) for c in record.param_diff] == [
        ("s3", "method", "log"),
        ("s4", "min_periods", 104),
    ]


@pytest.mark.parametrize(
    ("edits", "code"),
    [({"s4": {"min_periods": 1}}, "V03"), ({"s99": {"window": 3}}, None)],
    ids=["out-of-range value", "unknown step"],
)
def test_a_blocked_edit_ends_a_headless_run_and_nothing_runs(
    edits: dict[str, Any], code: str | None, settings: Settings
) -> None:
    graph = build_graph()
    run = start_run(graph, SUCCESS.request, list(SUCCESS.dataset_ids), settings)
    run = resume_run(graph, run, {"action": "edit", "edits": edits})
    assert run.paused_for is None and run.state["status"] == "blocked"
    assert run.state["report"].is_blocked and "steps" not in run.state
    if code:
        assert code in run.state["report"].codes()
    record = load_run(run.state["run_id"], settings.runs_dir)
    assert record.status == "blocked" and record.steps == [] and record.error


def test_rejecting_ends_the_run_unexecuted(settings: Settings) -> None:
    graph, run = start_review(SUCCESS.id, settings)
    run = resume_run(graph, run, {"action": "reject"})
    assert run.paused_for is None and run.state["status"] == "rejected"
    assert "steps" not in run.state and refusal_reason(run.state) == "The plan was rejected."
    events = read_events(settings.events_path)
    assert events[-1]["event_type"] == "plan_rejected"
    # The run has ended: a second Reject, or an Execute, changes nothing and logs nothing.
    for decision in (
        {"action": "reject"},
        {"action": "approve", "issue_ids": SCENARIO_5_APPROVALS},
    ):
        assert resume_run(graph, run, decision).state == run.state
    assert read_events(settings.events_path) == events
    record = load_run(run.state["run_id"], settings.runs_dir)
    assert (record.status, record.decision, record.error) == ("rejected", "rejected", None)
    assert record.plan_shown_at <= record.decided_at
    # A plan that never ran is not replayed (and logs no replay).
    report = replay(run.state["run_id"], settings)
    assert not report.all_match and "never executed" in report.message
    assert read_events(settings.events_path) == events
    assert activity(load_runs(settings.runs_dir))["rejected"] == 1


def test_a_refusal_has_no_plan_to_review(settings: Settings) -> None:
    graph, run = start_review("arima_forecast_block", settings)
    assert run.state["plan"].cannot_fulfill and run.state.get("plan_shown_at") is None
    refused = resume_run(graph, run, {"action": "reject"})
    assert refused.paused_for["error"] == "There is no plan to reject."


def test_a_bad_answer_is_refused_and_the_reviewer_is_asked_again(settings: Settings) -> None:
    graph = build_graph()
    run = start_run(graph, SUCCESS.request, list(SUCCESS.dataset_ids), settings)
    # An answer that does not fit ReviewDecision is refused before the node runs.
    with pytest.raises(ValidationError):
        resume_run(graph, run, {"action": "run it anyway"})
    # An issue that is not open: the run pauses again, with the reason.
    bad_ids = ["V10:s3:return_method", "V09:s5"]  # one open issue, one not of this plan
    run = resume_run(graph, run, {"action": "approve", "issue_ids": bad_ids})
    assert graph.get_state(run.config).next == ("approval",)
    assert "'V09:s5': not an open approval issue" in run.paused_for["error"]
    assert paused_ids(run) == SCENARIO_5_APPROVALS  # nothing was approved
    run = resume_run(graph, run, {"action": "approve", "issue_ids": SCENARIO_5_APPROVALS})
    assert run.state["status"] == "completed" and run.state["review_error"] is None
    granted = [
        e for e in read_events(settings.events_path) if e["event_type"] == "approval_granted"
    ]
    assert len(granted) == len(SCENARIO_5_APPROVALS) == len(run.state["approvals"])


# --- the request check and the planner -------------------------------------------


@pytest.mark.parametrize(
    "text", [" ".join(["word"] * (MAX_REQUEST_WORDS + 1)), "x" * (MAX_REQUEST_CHARS + 1)]
)
def test_r01_ends_the_run_before_the_planner(text: str, settings: Settings) -> None:
    client = FakeAnthropicClient(error=AssertionError("must not be called"))
    state = run_graph(text, ["NASDAQCOM"], "live", settings=settings, client=client)
    assert issue_codes(state) == ["R01"] and state["status"] == "blocked"
    assert client.calls == []  # the planner was never called: no tokens spent
    record = load_run(state["run_id"], settings.runs_dir)
    assert record.status == "blocked" and "1,000 words" in record.error
    # No LLM stage ran, so no tokens; the record names the planner the mode would use.
    assert record.stage_tokens == {} and record.plan_shown_at is None
    assert (record.llm.model, record.llm.prompt_version) == ("test-model", PROMPT_VERSION)
    events = read_events(settings.events_path)
    assert [e["details"]["code"] for e in events if e["event_type"] == "validation_issue"] == [
        "R01"
    ]


def test_a_request_at_the_limit_reaches_the_planner(settings: Settings) -> None:
    client = FakeAnthropicClient(error=RuntimeError("the planner was called"))
    text = " ".join(["word"] * MAX_REQUEST_WORDS)
    state = run_graph(text, ["NASDAQCOM"], "live", settings=settings, client=client)
    assert state["report"].codes() == ["V01"]  # the (failing) planner's issue, not R01
    assert len(client.calls) == 1


def test_a_planner_failure_is_a_recorded_v01(settings: Settings) -> None:
    client = FakeAnthropicClient(error=RuntimeError("service unavailable"))
    state = run_graph("anything", ["NASDAQCOM"], "live", settings=settings, client=client)
    assert state["report"].codes() == ["V01"] and state["status"] == "blocked"
    record = load_run(state["run_id"], settings.runs_dir)
    assert "service unavailable" in record.error
    assert record.llm.provider == "anthropic"


def test_an_unknown_mode_is_refused(settings: Settings) -> None:
    with pytest.raises(ValueError, match="unknown LLM mode"):
        start_run(build_graph(), SUCCESS.request, ["NASDAQCOM"], settings, mode="offline")


# --- live and cached runs ----------------------------------------------------------


def test_an_ungrounded_llm_narrative_is_withheld_and_recorded_as_n01(settings: Settings) -> None:
    # A live run: the (fake) planner returns example 5's plan, and the (fake) narrator
    # invents a number.
    invented = "Volatility peaked at 123.456 during the flagged weeks."
    client = FakeAnthropicClient([tool_response(raw_plan(SUCCESS.id)), text_response(invented)])
    state = run_graph(SUCCESS.request, list(SUCCESS.dataset_ids), "live", True, settings, client)

    # The numerical results stand: the run completes; the narrative shown is the template.
    assert state["status"] == "completed" and state["narrative"].source == "template"
    record = load_run(state["run_id"], settings.runs_dir)
    assert record.status == "completed"
    assert [(w.issue_id, w.action) for w in record.warnings] == [("N01:plan:llm", "WARN")]
    assert record.narrative["withheld"]["text"] == invented
    assert record.narrative["withheld"]["grounding"]["ungrounded"] == ["123.456"]
    assert record.narrative["note"] == ""  # no "every number checked" claim after N01
    events = read_events(settings.events_path)
    fired = [e["details"] for e in events if e["event_type"] == "guardrail_triggered"]
    assert [(d["code"], d["action"]) for d in fired] == [("N01", "WARN")]
    assert activity(load_runs(settings.runs_dir))["completed"] == 1


def test_a_live_run_records_the_tokens_of_each_stage(settings: Settings) -> None:
    # One fake answers both calls: a plan for the planner; for the narrator the same
    # answer has no text, so it is unusable, yet billed, and its tokens are recorded.
    client = FakeAnthropicClient(tool_response(raw_plan(SUCCESS.id)))
    state = run_graph(SUCCESS.request, list(SUCCESS.dataset_ids), "live", True, settings, client)
    assert state["status"] == "completed" and len(client.calls) == 2
    record = load_run(state["run_id"], settings.runs_dir)
    reported = TokenUsage(model="test-model", input_tokens=1200, output_tokens=300)
    assert record.stage_tokens == {"planning": reported, "narration": reported}
    assert record.narrative["source"] == "template"

    # The same request again in cached mode: the planner is not called; the cache
    # saved the original call's tokens.
    again = run_graph(SUCCESS.request, list(SUCCESS.dataset_ids), "cached", True, settings,
                      client)  # fmt: skip
    assert again["plan_result"].from_cache and len(client.calls) == 3  # the narrator only
    cached = load_run(again["run_id"], settings.runs_dir).stage_tokens["planning"]
    assert cached == TokenUsage(model="test-model", input_tokens=0, output_tokens=0,
                                saved_input_tokens=1200, saved_output_tokens=300)  # fmt: skip
