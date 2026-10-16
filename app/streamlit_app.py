"""QuantLab Streamlit app: entry point and page navigation.

Run with:

    uv run streamlit run app/streamlit_app.py

Pages (in app/pages/):
- Analysis: request -> plan -> validation and approvals -> execution -> results,
  narrative, trace and run record. Works fully in mock mode with no API key.
- Run history: past runs, their manifests, and replay with per-step hash checks.
- Monitoring: KPIs, latency per stage, capability usage, guardrail activity, errors.
- Capability registry: the approved menu of capabilities and the guardrail table.

The app is only a user interface: every action goes through quantlab.engine (the
same orchestrator the CLI and notebooks use).
"""

import streamlit as st

st.set_page_config(page_title="QuantLab", page_icon=":material/query_stats:", layout="wide")

pages = [
    st.Page("pages/analysis.py", title="Analysis", icon=":material/query_stats:", default=True),
    st.Page("pages/run_history.py", title="Run history", icon=":material/history:"),
    st.Page("pages/monitoring.py", title="Monitoring", icon=":material/monitoring:"),
    st.Page("pages/registry.py", title="Capability registry", icon=":material/menu_book:"),
]
st.navigation(pages).run()
