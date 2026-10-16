"""Plan schema: the inert data structure that the planner LLM produces.

Role in the architecture: a plan is DATA, not code. It is an ordered list of
steps; each step names a registered capability, says which earlier steps feed
its inputs, and gives parameter values. Nothing in a plan is ever evaluated:
the validator checks it field by field and the executor only uses the
capability name as a dictionary key.

`extra="forbid"` on both models means any field the schema does not define
(say, an LLM inventing `"code": "import os"`) makes the whole plan invalid (G01)
instead of being silently ignored.
"""

import copy
import hashlib
import json
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Step(BaseModel):
    """One step of a plan.

    Fields:
        id: step id, "s1", "s2", ... in order.
        capability: name of a registered capability.
        inputs: input name -> id of an EARLIER step whose output feeds it.
        params: parameter name -> value. Kept exactly as written (no coercion),
            so the validator can apply strict type rules.
        rationale: one sentence explaining why the step is there.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Step id: s1, s2, s3, ... in order.")
    capability: str = Field(description="Name of a registered capability.")
    inputs: dict[str, str] = Field(
        default_factory=dict,
        description="Input name -> id of an EARLIER step, e.g. {'series': 's1'}.",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameter name -> value, within the registry's types and ranges.",
    )
    rationale: str = Field(default="", description="One sentence: why this step.")


class Plan(BaseModel):
    """A complete plan for one user request.

    Fields:
        plan_id: unique id (set by the system, not the LLM).
        request: the user's request, verbatim (set by the system).
        dataset_ids: datasets the plan uses.
        steps: ordered steps.
        cannot_fulfill: True when the planner refuses (needs something not in the registry).
        refusal_reason: why the planner refused.
        planner_notes: free text from the planner; shown to the user, never trusted.
    """

    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(description="Unique plan id (set by the system).")
    request: str = Field(description="The user's request (set by the system).")
    dataset_ids: list[str] = Field(
        default_factory=list, description="Ids of the datasets the plan loads."
    )
    steps: list[Step] = Field(default_factory=list, description="Ordered plan steps.")
    cannot_fulfill: bool = Field(
        default=False,
        description="True if the request needs anything outside the registry or the rules.",
    )
    refusal_reason: str | None = Field(
        default=None, description="If cannot_fulfill: what is missing or not allowed."
    )
    planner_notes: str | None = Field(
        default=None, description="Short notes for the user (assumptions, caveats)."
    )


# Fields the system sets itself; the LLM is not asked to produce them.
SYSTEM_FIELDS = ("plan_id", "request")


def new_plan_id() -> str:
    """Return a fresh, short, unique plan id like "plan-1a2b3c4d5e"."""
    return f"plan-{uuid.uuid4().hex[:10]}"


def plan_hash(plan: Plan) -> str:
    """Version hash of a plan's content: dataset ids, steps and refusal fields.

    It changes whenever any parameter changes, so an approval stored with the hash
    of the plan it was given for cannot silently apply to an edited plan. The
    plan_id and request are excluded: they identify the run, not the computation.

    Args:
        plan: the plan.

    Returns:
        64-character hex digest.
    """
    content = plan.model_dump(exclude={"plan_id", "request"})
    text = json.dumps(content, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def plan_tool_schema() -> dict[str, Any]:
    """Return the JSON schema the planner LLM must fill in (its tool's input_schema).

    It is the Plan schema minus the system-set fields (plan_id, request), with
    the Step definition inlined so the schema is a single self-contained object.

    Returns:
        A JSON-schema dict.
    """
    full = Plan.model_json_schema()
    step_schema = copy.deepcopy(full["$defs"]["Step"])
    properties = {
        name: prop for name, prop in full["properties"].items() if name not in SYSTEM_FIELDS
    }
    properties["steps"] = {
        "type": "array",
        "items": step_schema,
        "description": full["properties"]["steps"]["description"],
    }
    return {
        "type": "object",
        "properties": properties,
        "required": ["dataset_ids", "steps", "cannot_fulfill"],
        "additionalProperties": False,
    }


def describe_step(step: Step) -> str:
    """One-line, plain description of a step, e.g. "s3 compute_returns(series=s2, method=simple)".

    Args:
        step: the step.

    Returns:
        The description.
    """
    args = [f"{name}={ref}" for name, ref in step.inputs.items()]
    args += [f"{name}={value}" for name, value in step.params.items()]
    return f"{step.id} {step.capability}({', '.join(args)})"


def describe_plan(plan: Plan) -> str:
    """Plain-text summary of a whole plan, one step per line (used by the narrator and CLI).

    Args:
        plan: the plan.

    Returns:
        Multi-line text.
    """
    if plan.cannot_fulfill:
        return f"Planner refused: {plan.refusal_reason}"
    return "\n".join(describe_step(step) for step in plan.steps)
