"""QuantLab Streamlit app: entry point and page navigation.

Run with:

    uv run streamlit run app/streamlit_app.py

Pages (in app/pages/):
- Analysis: request -> plan -> validation and approvals -> execution -> results,
  narrative, trace and run record. Works fully in mock mode with no API key.
- Run history: past runs, their manifests, and replay with per-step hash checks.
- Monitoring: KPIs, latency per stage, capability usage, guardrail activity, errors.
- Capability registry: the approved menu of capabilities.
- Guardrails: every guardrail, its action, who enforces it, its category and why it exists.

The sidebar shows the "QuantLab" title, then the page menu, then the page's own
controls. The app is only a user interface: every action goes through
quantlab.graph (the same orchestrator the CLI and notebooks use).
"""

import streamlit as st

st.set_page_config(page_title="QuantLab", page_icon=":material/query_stats:", layout="wide")

pages = [
    st.Page("pages/analysis.py", title="Analysis", icon=":material/query_stats:", default=True),
    st.Page("pages/run_history.py", title="Run history", icon=":material/history:"),
    st.Page("pages/monitoring.py", title="Monitoring", icon=":material/monitoring:"),
    st.Page("pages/registry.py", title="Capability registry", icon=":material/menu_book:"),
    st.Page("pages/guardrails.py", title="Guardrails", icon=":material/shield:"),
]
# The menu follows this list's order: st.navigation ignores file names, so the page
# files need no numeric prefixes.
#
# st.navigation always draws its own menu at the very top of the sidebar, so nothing
# could appear above it. Its menu is hidden, and the sidebar draws the title first,
# then one link per page from the same st.Page objects (same titles and icons).
# Each page's own sidebar content (e.g. Datasets on Analysis) follows when it runs.
current_page = st.navigation(pages, position="hidden")
with st.sidebar:
    st.header("QuantLab")  # the same call as the pages' sidebar headers, so the same style
    for page in pages:
        st.page_link(page, width="stretch")
current_page.run()
