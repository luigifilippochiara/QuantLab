"""Graph: the LangGraph prototype runs every example exactly as the engine does, pauses
at the approval interrupt and resumes from its checkpoint with the same state."""

import logging
from typing import Any

import pytest
from pydantic import ValidationError

from quantlab.config import Settings
from quantlab.engine import run_request
from quantlab.graph import (
    build_graph,
    current_plan_hash,
    issue_codes,
    resume_run,
    run_graph,
    start_run,
)
from quantlab.guardrails import MAX_REQUEST_WORDS
from quantlab.lineage import RunRecord, load_run, replay
from quantlab.mock_scenarios import SCENARIOS, SCENARIOS_BY_ID
from quantlab.monitoring import read_events
from quantlab.types import artifact_hash

from conftest import SCENARIO_5_APPROVALS

SUCCESS = SCENARIOS_BY_ID["buy_the_dip_tech"]  # scenario 5

# Fields that differ between two runs of the same request by nature: ids, clocks, latencies.
VOLATILE = {"run_id", "created_at", "plan_shown_at", "decided_at", "approved_at", "edited_at",
            "plan_id", "duration_ms", "latency_ms", "stage_timings_ms", "timestamp"}  # fmt: skip
# Record fields only the graph fills: engine.Session has no change requests.
GRAPH_ONLY = {"revisions", "manual_edits"}


def comparable(value: Any) -> Any:
    """A record or event as JSON, without its VOLATILE fields (at any depth)."""
    if isinstance(value, dict):
        return {k: comparable(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list):
        return [comparable(item) for item in value]
    return value


def run_events(settings: Settings, run_id: str) -> list[dict[str, Any]]:
    """The monitoring events of one run, in order, without their VOLATILE fields."""
    return [comparable(e) for e in read_events(settings.events_path) if e["run_id"] == run_id]


def paused_ids(run: Any) -> list[str]:
    """The issue ids a paused run asks the reviewer to approve."""
    return [issue["issue_id"] for issue in run.paused_for["open_approvals"]]


# Every example as the CLI demo runs it, and example 5 without its approvals.
CASES = [(s, s.auto_approve) for s in SCENARIOS] + [(SUCCESS, False)]


@pytest.mark.parametrize(
    ("scenario", "approve"), CASES, ids=[f"{s.id}-approve={a}" for s, a in CASES]
)
def test_every_example_runs_exactly_as_in_the_engine(
    scenario: Any, approve: bool, settings: Settings
) -> None:
    session = run_request(scenario.request, list(scenario.dataset_ids), "mock", approve, settings)
    state = run_graph(scenario.request, list(scenario.dataset_ids), "mock", approve, settings)
    assert state["status"] == session.status
    engine = load_run(session.run_id, settings.runs_dir)
    graph = load_run(state["run_id"], settings.runs_dir)
    # The whole run record: plan, issues, approvals, step hashes, narrative, error...
    assert comparable(graph.model_dump(mode="json", exclude=GRAPH_ONLY)) == comparable(
        engine.model_dump(mode="json", exclude=GRAPH_ONLY)
    )
    assert graph.stage_timings_ms.keys() == engine.stage_timings_ms.keys()
    # The graph's own fields: one revision (the planner was asked once), no change request.
    assert (graph.change_requests, graph.manual_edits) == (0, len(graph.param_diff))
    if graph.llm.error is None and graph.validation.codes() != ["R01"]:
        [revision] = graph.revisions
        assert revision.validation == engine.validation and revision.comment is None
    # ...and the same monitoring events, in the same order.
    assert run_events(settings, graph.run_id) == run_events(settings, engine.run_id)


def test_a_run_pauses_at_approval_and_resumes_with_the_same_state(settings: Settings) -> None:
    graph = build_graph()
    run = start_run(graph, SUCCESS.request, list(SUCCESS.dataset_ids), settings)
    snapshot = graph.get_state(run.config)
    assert snapshot.next == ("approval",) and paused_ids(run) == SCENARIO_5_APPROVALS
    # The checkpoint gives back the very same state, rebuilt from its stored form.
    assert snapshot.values == run.state and run.state["status"] == "ready"
    # A run awaiting a decision is on record, as in the engine.
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
def test_a_blocked_edit_ends_the_run_and_nothing_runs(
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
    graph = build_graph()
    run = start_run(graph, SUCCESS.request, list(SUCCESS.dataset_ids), settings)
    run = resume_run(graph, run, {"action": "reject"})
    assert run.state["status"] == "rejected" and "steps" not in run.state
    record = load_run(run.state["run_id"], settings.runs_dir)
    assert (record.status, record.decision, record.error) == ("rejected", "rejected", None)
    assert read_events(settings.events_path)[-1]["event_type"] == "plan_rejected"


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


def test_r01_ends_the_run_before_the_planner(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_planner(*args: Any) -> None:
        raise AssertionError("the planner must not be called")

    monkeypatch.setattr("quantlab.graph.generate_plan", no_planner)
    text = " ".join(["word"] * (MAX_REQUEST_WORDS + 1))
    state = run_graph(text, ["NASDAQCOM"], "mock", False, settings)
    assert issue_codes(state) == ["R01"] and state["status"] == "blocked"
    record = load_run(state["run_id"], settings.runs_dir)
    assert "1,000 words" in record.error and record.stage_tokens == {}


def test_an_unknown_mode_is_refused(settings: Settings) -> None:
    with pytest.raises(ValueError, match="unknown LLM mode"):
        start_run(build_graph(), SUCCESS.request, ["NASDAQCOM"], settings, mode="offline")


def test_every_checkpointed_class_is_allowlisted(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    # A class missing from CHECKPOINT_TYPES is logged when a run resumes, and comes
    # back as a plain dict (or, for an enum, a plain string).
    caplog.set_level(logging.WARNING)
    for scenario in SCENARIOS:
        run_graph(scenario.request, list(scenario.dataset_ids), "mock", True, settings)
    assert not [r.getMessage() for r in caplog.records if r.name.startswith("langgraph")]
