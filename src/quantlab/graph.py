"""Graph: the orchestrator used by the Streamlit app, the CLI and the notebooks.

Role in the architecture: a LangGraph `StateGraph` moves one run through the
pipeline, from request to saved record. This module wires it: the edges, the
checkpointer, the compiled graph and the run API. The run's state and the nodes
that change it live in nodes.py. Every run is saved (lineage.save_run), so it can
be replayed and traced.

    START -> request_check -> planner -> validator -> approval -> executor -> narrator
          -> recorder -> END

    request_check  R01 refuses an oversized request (or change request) before any LLM call
    planner        the planner proposes a plan (or refuses, P01), or revises it
    validator      the validator's guardrails; run again after every edit and revision
    approval       interrupt(): the run pauses until a reviewer approves, edits, rejects
                   or requests changes
    executor       deterministic execution with the runtime guardrails
    narrator       narrative and grounding check (an ungrounded one is withheld, N01)
    recorder       saves the run record; every run ends here, blocked or not

Conditional edges: a terminal block (V01, V02) goes to the recorder and ends. For a
person reviewing (interactive, the Analysis page) every other plan pauses at the
approval interrupt, blocked or not, so it can be edited, executed, rejected or sent
back with a change request; headless, a BLOCK ends the run, open APPROVE issues
pause it and a clean plan runs. An edit goes back to the validator; a change request
(live mode only, at most MAX_CHANGE_REQUESTS per run) goes back through the request
check and the planner, then the validator: nothing skips validation. Every revision
stays under the one run id. The state is checkpointed after every node by an
in-memory checkpointer, so a paused run resumes with exactly the same state.

The module reads top to bottom: the edges (the one after the validator is in
nodes.py, since validator_node consults it too); building the graph (the
checkpointer and the wiring); then the run API: `start_run` and `resume_run` (the
Analysis page answers the interrupt) and `run_graph`, which chains them for headless
use (CLI, notebooks, tests).
"""

from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from quantlab.config import LLM_MODES, Settings, load_settings
from quantlab.data import load_catalog
from quantlab.executor import StepRecord
from quantlab.guardrails import Action, Category, GuardrailType, Issue, ValidationReport
from quantlab.lineage import Approval, ParamEdit, PlanRevision, RunRecord, new_run_id
from quantlab.monitoring import utc_now
from quantlab.narrator import NarrativeResult
from quantlab.nodes import (
    GraphContext,
    GraphState,
    after_validation,
    approval_node,
    executor_node,
    narrator_node,
    planner_node,
    recorder_node,
    request_check_node,
    validator_node,
)
from quantlab.plan import Plan
from quantlab.planner import PlanResult, RevisionRequest
from quantlab.types import DataType, Frequency, SeriesKind, TokenUsage

# ---------------------------------------------------------------------------
# Edges: where each node's result leads
# ---------------------------------------------------------------------------


def after_request_check(state: GraphState) -> str:
    """An accepted change request goes to the planner, a refused one back to the
    reviewer; R01 on the request ends the run; any other request goes to the planner."""
    if state.get("revision_request") is not None:
        return "planner"
    if state.get("review") == "refused":
        return "approval"
    return "recorder" if state.get("status") == "blocked" else "planner"


def after_planner(state: GraphState) -> str:
    """No plan at all (V01, at any revision) ends the run; a plan or a refusal (P01) is
    validated."""
    return "validator" if state.get("plan") is not None else "recorder"


# after_validation, the edge after the validator, is in nodes.py: validator_node
# consults it to save a run that will pause for review.


def after_review(state: GraphState) -> str:
    """An edit is validated again; a change request goes through the request check to
    the planner; full approval runs the plan; a refused answer asks again; a rejection,
    or (headless) approval not given for every issue, ends the run."""
    if state["review"] == "edited":
        return "validator"
    if state["review"] == "changes_requested":
        return "request_check"
    if state["review"] == "approved":
        return "executor"
    if state["review"] == "refused":
        return "approval"
    return "recorder"


def after_execution(state: GraphState) -> str:
    """A completed run is narrated; a runtime BLOCK or a failure ends the run."""
    return "narrator" if state["status"] == "completed" else "recorder"


# ---------------------------------------------------------------------------
# Building the graph: the checkpointer and the wiring
# ---------------------------------------------------------------------------


# The classes the checkpointer may rebuild from a checkpoint: the types held in
# GraphState and the enums inside their dumps (an Issue's dump includes its
# computed type and category). A class missing here would come back as a plain
# dict or string, so tests/test_graph.py checks the whole state round-trips.
CHECKPOINT_TYPES = [
    PlanResult,
    Plan,
    Issue,
    Action,
    GuardrailType,
    Category,
    ValidationReport,
    ParamEdit,
    Approval,
    StepRecord,
    NarrativeResult,
    RunRecord,
    PlanRevision,
    RevisionRequest,
    DataType,
    Frequency,
    SeriesKind,
    TokenUsage,
]


def new_checkpointer() -> InMemorySaver:
    """An in-memory checkpointer that can hold QuantLab's state.

    Pydantic models and dataclasses are stored as msgpack and rebuilt on load;
    the allowlist limits rebuilding to CHECKPOINT_TYPES. Artifacts hold pandas
    Series, which msgpack cannot encode, so that one value falls back to pickle.
    The checkpoints only ever live in this process's memory, so no pickle from
    outside is ever loaded.
    """
    serde = JsonPlusSerializer(pickle_fallback=True, allowed_msgpack_modules=CHECKPOINT_TYPES)
    return InMemorySaver(serde=serde)


def build_graph(checkpointer: InMemorySaver | None = None) -> CompiledStateGraph:
    """Wire the nodes and edges and compile the graph.

    Args:
        checkpointer: where to keep each run's state between nodes (default: a new
            in-memory one). Required by the approval interrupt.

    Returns:
        The compiled graph; one graph can hold many runs (one thread per run id).
    """
    builder = StateGraph(GraphState, context_schema=GraphContext)
    builder.add_node("request_check", request_check_node)
    builder.add_node("planner", planner_node)
    builder.add_node("validator", validator_node)
    builder.add_node("approval", approval_node)
    builder.add_node("executor", executor_node)
    builder.add_node("narrator", narrator_node)
    builder.add_node("recorder", recorder_node)

    builder.add_edge(START, "request_check")
    builder.add_conditional_edges(
        "request_check", after_request_check, ["planner", "approval", "recorder"]
    )
    builder.add_conditional_edges("planner", after_planner, ["validator", "recorder"])
    builder.add_conditional_edges(
        "validator", after_validation, ["approval", "executor", "recorder"]
    )
    builder.add_conditional_edges(
        "approval",
        after_review,
        ["validator", "request_check", "executor", "approval", "recorder"],
    )
    builder.add_conditional_edges("executor", after_execution, ["narrator", "recorder"])
    builder.add_edge("narrator", "recorder")
    builder.add_edge("recorder", END)
    return builder.compile(checkpointer=checkpointer or new_checkpointer())


# ---------------------------------------------------------------------------
# Running the graph
# ---------------------------------------------------------------------------


@dataclass
class GraphRun:
    """Where a run stands when the graph returns: finished, or paused for review.

    Fields:
        config: the checkpointer config of the run (its thread id is the run id).
        context: settings, catalog and LLM client, passed to the graph again on resume.
        state: the run's state so far.
        paused_for: the approval interrupt's payload while paused, else None.
    """

    config: dict[str, Any]
    context: GraphContext
    state: GraphState = field(default_factory=dict)
    paused_for: dict[str, Any] | None = None


def _invoke(graph: CompiledStateGraph, run: GraphRun, graph_input: Any) -> GraphRun:
    """Run the graph until it ends or pauses, and report where it stands."""
    output = graph.invoke(graph_input, run.config, context=run.context, version="v2")
    paused_for = output.interrupts[0].value if output.interrupts else None
    return GraphRun(run.config, run.context, output.value, paused_for)


def start_run(
    graph: CompiledStateGraph,
    request: str,
    dataset_ids: list[str],
    settings: Settings,
    mode: str = "mock",
    interactive: bool = False,
    client: Any = None,
) -> GraphRun:
    """Start a run: it goes as far as it can, then ends or pauses at the approval interrupt.

    Args:
        graph: a graph from build_graph().
        request: the user's request.
        dataset_ids: datasets selected by the user.
        settings: settings in force.
        mode: LLM mode: "mock" (the default), "live" or "cached"; change requests
            need "live".
        interactive: True when a person reviews the plan (the Analysis page).
        client: optional OpenAI or Anthropic client (tests inject a fake one).

    Returns:
        The GraphRun; `paused_for` is set while the run waits for a review.

    Raises:
        ValueError: for an unknown mode.
    """
    if mode not in LLM_MODES:
        raise ValueError(f"unknown LLM mode {mode!r}; expected one of {LLM_MODES}")
    run_id = new_run_id()
    state: GraphState = {
        "request": request,
        "dataset_ids": list(dataset_ids),
        "mode": mode,
        "run_id": run_id,
        "created_at": utc_now(),
        "interactive": interactive,
        "status": "planned",
    }
    catalog = load_catalog(settings.catalog_path)
    context = GraphContext(settings=settings, catalog=catalog, client=client)
    run = GraphRun(config={"configurable": {"thread_id": run_id}}, context=context)
    return _invoke(graph, run, state)


def resume_run(graph: CompiledStateGraph, run: GraphRun, decision: dict[str, Any]) -> GraphRun:
    """Answer the approval interrupt of a paused run and let it go on from the checkpoint.

    Args:
        graph: the graph the run was started on (its checkpointer holds the state).
        run: the paused run.
        decision: the reviewer's answer, in ReviewDecision's fields, e.g.
            {"action": "approve", "issue_ids": ["V10:s3:return_method"]}.

    Returns:
        The GraphRun after the decision: finished, or paused again (after an edit, a
        change request or a refused answer).
    """
    return _invoke(graph, run, Command(resume=decision))


def run_graph(
    request: str,
    dataset_ids: list[str],
    mode: str | None = None,
    approve: bool = False,
    settings: Settings | None = None,
    client: Any = None,
    graph: CompiledStateGraph | None = None,
) -> GraphState:
    """Run the whole graph headlessly (CLI, notebooks, tests): no person reviews the plan.

    Args:
        request: the user's request.
        dataset_ids: selected datasets.
        mode: LLM mode (default: the configured one).
        approve: approve every open APPROVE issue at the interrupt (CLI --approve-all;
            else none is, and the run is recorded as blocked, nothing executed).
        settings: settings (default load_settings()).
        client: optional OpenAI or Anthropic client (tests inject a fake one).
        graph: a graph to run on (default: a new one).

    Returns:
        The final state. Its status says how far the run got: "blocked",
        "failed" or "completed"; its record is the saved run record.
    """
    settings = settings or load_settings()
    graph = graph or build_graph()
    run = start_run(graph, request, dataset_ids, settings, mode or settings.llm_mode, client=client)
    if run.paused_for is not None:
        open_ids = [issue["issue_id"] for issue in run.paused_for["open_approvals"]]
        decision = {"action": "approve", "issue_ids": open_ids if approve else []}
        run = resume_run(graph, run, {**decision, "approved_by": "cli"})
    return run.state
