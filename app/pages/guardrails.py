"""Guardrails page: every static and runtime guardrail, with what goes wrong without it.

Everything shown is generated from quantlab.guardrails.GUARDRAILS, the table the
validator and executor take each issue's name and severity from, so this page
cannot drift from what they enforce.
"""

import pandas as pd
import streamlit as st

from quantlab.guardrails import GUARDRAILS

static_count = sum(g.stage == "static" for g in GUARDRAILS.values())
runtime_count = sum(g.stage == "runtime" for g in GUARDRAILS.values())

st.title("Guardrails")
st.caption(
    f"The safety net: every plan passes {static_count} static guardrails, enforced "
    f"deterministically before anything runs, and {runtime_count} runtime guardrails, checked "
    "on the data during execution. A BLOCK stops the run, a WARN is shown to you, and an "
    "APPROVAL needs your sign-off."
)

# A static table (not an interactive dataframe) so the long explanations wrap and all rows show.
st.table(
    pd.DataFrame(
        [
            {
                "code": g.code,
                "name": g.name,
                "severity": g.severity.value,
                "stage": g.stage,
                "what goes wrong without it": g.risk,
            }
            for g in GUARDRAILS.values()
        ]
    ).set_index("code")
)
