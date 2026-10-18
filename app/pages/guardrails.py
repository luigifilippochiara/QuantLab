"""Guardrails page: every guardrail, who enforces it, what it does when it fires, and why.

Everything shown is generated from quantlab.guardrails.GUARDRAILS, the table the
planner refusal, the validator, the executor and the narrator check take each
issue's name and action from, so this page cannot drift from what they enforce.
The filter options come from the same table (in the order of the Action,
GuardrailType and Category enums and of ENFORCERS), so they always match the
values it holds; the Category filter's tooltip gives each category's definition
(CATEGORY_DEFINITIONS).
"""

import pandas as pd
import streamlit as st

from quantlab.guardrails import (
    CATEGORY_DEFINITIONS,
    ENFORCERS,
    GUARDRAILS,
    Action,
    Category,
    GuardrailType,
)

INTRO = (
    "Guardrails are the checks that keep an analysis safe and meaningful. When one fires, "
    "its action decides what happens: **BLOCK** stops the run, **APPROVE** needs a human "
    "decision, **WARN** lets the run continue and flags the result."
)

ROWS = [
    {
        "Code": g.code,
        "Name": g.name,
        "Action": g.action.value,
        "Enforced by": g.enforced_by,
        "Type": g.type.value,
        "Category": g.category.value,
        "Description": g.description,
    }
    for g in GUARDRAILS.values()
]
# Filter options: only the values the table uses, in declaration order.
ACTION_OPTIONS = [a.value for a in Action if any(row["Action"] == a for row in ROWS)]
ENFORCER_OPTIONS = [e for e in ENFORCERS.values() if any(row["Enforced by"] == e for row in ROWS)]
TYPE_OPTIONS = [t.value for t in GuardrailType if any(row["Type"] == t for row in ROWS)]
CATEGORY_OPTIONS = [c.value for c in Category if any(row["Category"] == c for row in ROWS)]
# Widget key of each multiselect -> the table column it filters.
FILTERS = {
    "guardrail_action": "Action",
    "guardrail_enforcer": "Enforced by",
    "guardrail_type": "Type",
    "guardrail_category": "Category",
}


def clear_filters() -> None:
    """Reset the search and every filter (a button callback: it runs before the widgets)."""
    st.session_state.guardrail_search = ""
    for key in FILTERS:
        st.session_state[key] = []


def matches(row: dict[str, str], search: str, selected: dict[str, list[str]]) -> bool:
    """Whether a table row passes the search and every filter (AND; empty means all).

    Args:
        row: one table row.
        search: text matched, case-insensitively and partially, against code and name.
        selected: table column -> the values chosen in its filter.
    """
    text = search.strip().lower()
    if text and text not in row["Code"].lower() and text not in row["Name"].lower():
        return False
    return all(not values or row[column] in values for column, values in selected.items())


st.title("Guardrails")
st.caption(INTRO)

search_col, action_col, enforcer_col, type_col, category_col = st.columns([2, 1, 1.2, 1, 1.3])
search = search_col.text_input(
    "Search code or name", key="guardrail_search", placeholder="e.g. V0, look, P01"
)
action_col.multiselect("Action", ACTION_OPTIONS, key="guardrail_action")
enforcer_col.multiselect("Enforced by", ENFORCER_OPTIONS, key="guardrail_enforcer")
type_col.multiselect("Type", TYPE_OPTIONS, key="guardrail_type")
category_col.multiselect(
    "Category",
    CATEGORY_OPTIONS,
    key="guardrail_category",
    help="\n\n".join(f"**{c}**: {CATEGORY_DEFINITIONS[Category(c)]}" for c in CATEGORY_OPTIONS),
)

selected = {column: st.session_state[key] for key, column in FILTERS.items()}
shown = [row for row in ROWS if matches(row, search, selected)]
filtered = bool(search.strip()) or any(selected.values())
count_col, clear_col = st.columns([4, 1], vertical_alignment="center")
count_col.caption(f"Showing {len(shown)} of {len(ROWS)} guardrails")
clear_col.button(
    "Clear filters",
    on_click=clear_filters,
    icon=":material/filter_alt_off:",
    disabled=not filtered,
    width="stretch",
)

# A static table (not an interactive dataframe) so the long descriptions wrap and all rows show.
if shown:
    st.table(pd.DataFrame(shown).set_index("Code"))
else:
    st.info("No guardrail matches these filters.", icon=":material/filter_alt_off:")
