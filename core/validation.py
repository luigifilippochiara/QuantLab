"""Plan validation: check a plan against the registry before anything runs.

A plan is plain data (a dict, usually loaded from JSON), for example:

    {
        "plan_id": "example-rolling-vol-001",
        "description": "20-day rolling volatility of the sample prices.",
        "steps": [
            {
                "step_id": "s1",
                "capability": "rolling_volatility",
                "input": "prices",
                "params": {"window": 20, "periods_per_year": 252},
                "output": "vol_20d"
            }
        ]
    }

This first version checks that:
  1. the plan has the right structure (the Plan / PlanStep models below);
  2. each step's capability exists in the registry;
  3. each step's input exists and has the type the capability expects;
  4. every parameter is present, known, of the right type, and within range.

It returns *all* problems at once instead of stopping at the first one, so a
future planner can send the full list back to the LLM and ask for a fix.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from core.registry import REGISTRY, CapabilitySpec, DataType, ParamType


class PlanStep(BaseModel):
    """One step of a plan: run `capability` on `input` with `params`, store as `output`.

    Fields:
        step_id: short identifier for messages and logs, e.g. "s1".
        capability: name of a registered capability.
        input: name of the data to read (a supplied dataset or an earlier step's output).
        params: parameter name -> value. Checked against the registry by `check_params`.
        output: name under which this step's result is stored.
    """

    # extra="forbid": unexpected keys (e.g. an LLM inventing "code": "...") are rejected.
    model_config = ConfigDict(extra="forbid")

    step_id: str
    capability: str
    input: str
    # Values are kept exactly as written (Any) so `check_params` can apply strict
    # type rules, rather than letting Pydantic quietly convert "20" into 20.
    params: dict[str, Any]
    output: str


class Plan(BaseModel):
    """A full plan: an ordered list of steps, run top to bottom.

    Fields:
        plan_id: identifier for logging and reproducibility.
        description: plain-language summary of what the plan does.
        steps: at least one PlanStep.
    """

    model_config = ConfigDict(extra="forbid")

    plan_id: str
    description: str
    steps: list[PlanStep] = Field(min_length=1)


def _has_expected_type(value: Any, expected: ParamType) -> bool:
    """Check whether a parameter value matches its declared type.

    Args:
        value: the value from the plan.
        expected: the declared type, "int" or "float".

    Returns:
        True if the value is acceptable for that type, otherwise False.
    """
    # In Python, bool is a subclass of int (True == 1), so reject it explicitly.
    # Otherwise "window": true in a plan would silently mean window=1.
    if isinstance(value, bool):
        return False
    if expected == "int":
        return isinstance(value, int)
    if expected == "float":
        # Whole numbers are fine for a float parameter (2 means 2.0), but not the reverse.
        return isinstance(value, (int, float))
    return False


def check_params(spec: CapabilitySpec, params: dict[str, Any]) -> list[str]:
    """Check a step's parameters against a capability's spec.

    Used by `validate_plan` and again by the execution engine right before it
    runs a capability, so both apply exactly the same rules.

    Args:
        spec: the capability's spec from the registry.
        params: parameter name -> value, as written in the plan.

    Returns:
        A list of error messages; empty if all parameters are valid.
    """
    errors: list[str] = []

    # Every parameter must be given explicitly. There are no hidden defaults, so
    # the plan on its own fully describes the computation that will be run.
    for name in spec.params:
        if name not in params:
            errors.append(f"missing parameter '{name}'.")

    for name, value in params.items():
        param_spec = spec.params.get(name)
        if param_spec is None:
            errors.append(
                f"unknown parameter '{name}' for '{spec.name}'; "
                f"allowed: {sorted(spec.params)}."
            )
            continue

        if not _has_expected_type(value, param_spec.type):
            errors.append(
                f"parameter '{name}' must be of type {param_spec.type}, "
                f"got {value!r} ({type(value).__name__})."
            )
            continue  # a range check makes no sense on a value of the wrong type

        # Inclusive range check. A NaN value fails this too, since any comparison with NaN is False.
        if not (param_spec.min_value <= value <= param_spec.max_value):
            errors.append(
                f"parameter '{name}'={value} is outside the allowed range "
                f"[{param_spec.min_value:g}, {param_spec.max_value:g}]."
            )

    return errors


def validate_plan(raw_plan: dict, available_inputs: dict[str, DataType]) -> list[str]:
    """Check a whole plan against the registry without running anything.

    Args:
        raw_plan: the plan as a dict (e.g. loaded from JSON).
        available_inputs: datasets the caller will supply, mapped to their data
            type, e.g. {"prices": "price_series"}.

    Returns:
        A list of human-readable error messages. An empty list means the plan
        is valid and safe to pass to the execution engine.
    """
    # 1. Structure: parse the dict into the Plan model.
    try:
        plan = Plan.model_validate(raw_plan)
    except ValidationError as exc:
        # Turn each Pydantic error into one readable line. err["loc"] is the path to
        # the problem, e.g. ("steps", 0, "output") -> "steps.0.output: Field required".
        structure_errors = []
        for err in exc.errors():
            location = ".".join(str(part) for part in err["loc"]) or "(whole plan)"
            structure_errors.append(f"plan structure: {location}: {err['msg']}")
        return structure_errors

    errors: list[str] = []

    # Every named piece of data a step may read, with its type: the caller's
    # inputs first, then each step's output as we walk through the plan in order.
    # This is what will let a later step use an earlier step's result.
    known_data: dict[str, DataType] = dict(available_inputs)

    for step in plan.steps:
        label = f"step '{step.step_id}'"

        # 2. The capability must exist in the registry.
        spec = REGISTRY.get(step.capability)
        if spec is None:
            errors.append(
                f"{label}: unknown capability '{step.capability}'; "
                f"allowed: {sorted(REGISTRY)}."
            )
            continue  # without a spec we can't check this step's input or params

        # 3. The input must exist and be the type this capability expects.
        if step.input not in known_data:
            errors.append(
                f"{label}: input '{step.input}' does not exist; "
                f"available: {sorted(known_data)}."
            )
        elif known_data[step.input] != spec.input_type:
            errors.append(
                f"{label}: '{step.capability}' expects {spec.input_type}, "
                f"but '{step.input}' is {known_data[step.input]}."
            )

        # 4. Parameters, using the same rules the engine applies before running.
        errors.extend(f"{label}: {msg}" for msg in check_params(spec, step.params))

        # Register this step's output so later steps can use it as their input.
        # Refuse to overwrite existing data, which would make results hard to trace.
        if step.output in known_data:
            errors.append(f"{label}: output name '{step.output}' is already in use.")
        else:
            known_data[step.output] = spec.output_type

    return errors
