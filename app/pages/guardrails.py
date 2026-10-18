"""Guardrails page: every guardrail, who enforces it, what it does when it fires, and why.

Everything shown is generated from quantlab.guardrails.GUARDRAILS, the table the
planner refusal, the validator, the executor and the narrator check take each
issue's name and action from, so this page cannot drift from what they enforce.
"Enforced by" is written out (it is derived from the code's prefix), so the
reader never has to decode L, G, R or N.
"""

import pandas as pd
import streamlit as st

from quantlab.guardrails import ENFORCERS, GUARDRAILS, Action

# Who enforces how many guardrails, in pipeline order (planner first, narrator check last).
enforcer_counts = {
    enforcer: sum(g.enforced_by == enforcer for g in GUARDRAILS.values())
    for enforcer in ENFORCERS.values()
}
ACTION_MEANINGS = {
    Action.BLOCK: "the run stops",
    Action.APPROVE: "a human must decide before anything runs",
    Action.WARN: "the run continues and the result is flagged",
}

who = ", ".join(f"{count} enforced by the {name}" for name, count in enforcer_counts.items())

st.title("Guardrails")
st.caption(
    f"Every guardrail QuantLab applies, in pipeline order: {who}. "
    "When one fires, its action says what happens: "
    + "; ".join(f"**{action.value}**: {meaning}" for action, meaning in ACTION_MEANINGS.items())
    + "."
)

# A static table (not an interactive dataframe) so the long descriptions wrap and all rows show.
st.table(
    pd.DataFrame(
        [
            {
                "Code": g.code,
                "Name": g.name,
                "Action": g.action.value,
                "Enforced by": g.enforced_by,
                "Category": g.category.value,
                "Description": g.description,
            }
            for g in GUARDRAILS.values()
        ]
    ).set_index("Code")
)
