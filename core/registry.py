"""Capability registry: the single source of truth for what the engine may run.

A *capability* is one named, pre-written analysis function, such as
`rolling_volatility`. The registry describes each capability as data: its
parameters (with types and allowed ranges), what kind of data it takes in, and
what kind it gives back. It contains no executable logic.

Who reads it:
- validation checks plans against it;
- execution refuses to run anything that is not in it;
- the planner (future) will show it to the LLM, so the LLM knows exactly what
  it is allowed to ask for.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

# The kinds of data that flow between plan steps. Each capability declares the
# kind it accepts and the kind it produces, so a multi-step plan can be
# type-checked before running (e.g. "this step expects prices, not volatilities").
DataType = Literal["price_series", "volatility_series"]

# Parameter types supported so far. Add more (e.g. "choice") when a capability needs them.
ParamType = Literal["int", "float"]


class ParamSpec(BaseModel):
    """Specification of one parameter of a capability.

    Fields:
        type: "int" or "float".
        min_value: smallest allowed value (inclusive).
        max_value: largest allowed value (inclusive).
        description: what the parameter means, in plain words.
    """

    # frozen=True: fields can't be reassigned after creation, so a spec isn't
    # accidentally loosened at runtime. extra="forbid": a misspelt field name
    # (e.g. "max_val") raises an error instead of being silently ignored.
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: ParamType
    # Bounds are required on purpose: every parameter must have an explicit allowed range.
    min_value: float
    max_value: float
    description: str


class CapabilitySpec(BaseModel):
    """Specification of one capability the engine is allowed to run.

    Fields:
        name: unique identifier used in plans, e.g. "rolling_volatility".
        description: short plain-language summary of what it computes.
        input_type: kind of data the capability consumes.
        output_type: kind of data it produces.
        params: parameter name -> ParamSpec. Every parameter is required.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str
    input_type: DataType
    output_type: DataType
    params: dict[str, ParamSpec]


ROLLING_VOLATILITY = CapabilitySpec(
    name="rolling_volatility",
    description=(
        "Annualised rolling volatility: the standard deviation of log returns "
        "over a trailing window, scaled by sqrt(periods_per_year)."
    ),
    input_type="price_series",
    output_type="volatility_series",
    params={
        "window": ParamSpec(
            type="int",
            # 2 is the technical minimum: a sample standard deviation needs at least
            # 2 observations. Small windows are allowed but give very noisy estimates.
            min_value=2,
            # 252 is about one trading year. This is a sanity cap, not a law of nature.
            max_value=252,
            description="Number of returns in each rolling window.",
        ),
        "periods_per_year": ParamSpec(
            type="int",
            # 1 means "don't annualise" (report per-period volatility).
            # 365 covers assets that trade every calendar day (e.g. crypto).
            min_value=1,
            max_value=365,
            description=(
                "Return periods per year, used to annualise: 252 for daily trading "
                "data, 52 for weekly, 12 for monthly, 1 for no annualisation."
            ),
        ),
    },
)


# The registry itself: capability name -> spec.
# Built from a list so each key is guaranteed to equal the spec's own `name`.
REGISTRY: dict[str, CapabilitySpec] = {spec.name: spec for spec in [ROLLING_VOLATILITY]}
