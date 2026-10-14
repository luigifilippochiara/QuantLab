"""Planner (stub — not implemented yet).

Future role: take a natural-language request ("show me 20-day volatility of
these prices") and ask an LLM to return a *plan*: structured JSON that only
names capabilities from the registry and gives values for their parameters.

The LLM never writes or runs code. Its output is data, which is then checked
by `core.validation` and run by `core.execution`. If validation fails, the
error messages can be sent back to the LLM so it can propose a corrected plan.
"""
