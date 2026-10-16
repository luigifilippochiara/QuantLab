"""Capability registry page: the approved menu, plus the guardrail table.

Everything shown is generated from quantlab.registry and quantlab.guardrails,
so this page cannot drift from what the validator and executor enforce.
"""

import pandas as pd
import streamlit as st

from quantlab.guardrails import GUARDRAILS
from quantlab.registry import REGISTRY, ParamSpec, llm_summary, registry_version

import ui


def allowed_values(p: ParamSpec) -> str:
    """What a parameter accepts, in a few words: its choices, its range or its date format."""
    if p.choices:
        return ", ".join(p.choices)
    if p.min is not None:
        return f"{p.min:g} to {p.max:g}"
    return "YYYY-MM-DD" if p.type == "date" else "-"


st.title("Capability registry")
st.caption(
    f"The approved menu: the planner can only name these {len(REGISTRY)} capabilities, and the "
    f"executor can only run them. Registry version `{registry_version()[:16]}`."
)

for spec in REGISTRY.values():
    with st.expander(f"{spec.name} → {spec.output_type.value}", icon=":material/function:"):
        st.markdown(ui.escape_md(spec.description))
        if spec.inputs:
            st.markdown("**Inputs** (references to earlier steps)")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "input": i.name,
                            "data types": ", ".join(t.value for t in i.data_types),
                            "allowed kinds": ", ".join(k.value for k in i.kinds)
                            if i.kinds
                            else "any",
                            "description": i.description,
                        }
                        for i in spec.inputs
                    ]
                ),
                hide_index=True,
                alt=f"{spec.name} inputs",
            )
        if spec.params:
            st.markdown("**Parameters**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "parameter": p.name,
                            "type": p.type,
                            "required": p.required,
                            "default": "-"
                            if p.required
                            else ("null (not set)" if p.default is None else str(p.default)),
                            "allowed": allowed_values(p),
                            "description": p.description,
                        }
                        for p in spec.params
                    ]
                ),
                hide_index=True,
                alt=f"{spec.name} parameters",
            )
        output_kind = spec.output_kind or "-"
        st.markdown(f"**Output:** {spec.output_type.value} · kind `{output_kind}`")
        st.markdown(f"**Minimum observations:** {ui.escape_md(spec.min_obs.description)}")
        st.markdown("**Assumptions**")
        for assumption in spec.assumptions:
            flag = " :orange-badge[needs approval]" if assumption.requires_approval else ""
            st.markdown(f"- {ui.escape_md(assumption.text)}{flag}")
        st.markdown("**Underlying library calls**")
        st.code(spec.library_calls, language="python")

st.subheader("Guardrails")
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

with st.expander("What the planner LLM sees (generated summary)", icon=":material/smart_toy:"):
    st.code(llm_summary(), language="json")
