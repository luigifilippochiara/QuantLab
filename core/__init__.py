"""QuantLab core package.

Pipeline (left to right):

    request --> planner --> plan --> validation --> execution --> result
                                         ^              ^
                                         |              |
                                     registry  (the only source of truth
                                                about what is allowed)

- registry:   declares which capabilities exist and what parameters they accept.
- planner:    (future) turns a natural-language request into a plan, via an LLM.
- validation: checks a plan against the registry before anything runs.
- execution:  deterministic engine that runs a validated plan step by step.
- data:       (future) loading and checking input data.
- monitoring: (future) logging and auditing of plans and runs.
"""
