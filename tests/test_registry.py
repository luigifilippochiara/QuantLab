"""The registry is internally consistent and its LLM summary is generated, not hand-written.

Nothing here counts the capabilities: the checks hold for any registry, so adding
or retiring a capability needs no test edits beyond its own (and the code ledger).
"""

import inspect
import json
import re

from quantlab import capabilities
from quantlab.registry import REGISTRY, capability_code, llm_summary, registry_version

# Every capability code ever assigned. Codes are permanent: a code is never
# renumbered or reused. Add a new capability's code here; when a capability is
# retired, move its entry to RETIRED_CODES instead of deleting it.
ASSIGNED_CODES = {
    "C01": "load_series",
    "C02": "resample",
    "C03": "compute_returns",
    "C04": "difference",
    "C05": "spread",
    "C06": "rolling_volatility",
    "C07": "flag_regime",
    "C08": "summary_stats",
    "C09": "compare_groups",
    "C10": "filter_period",
    "C11": "find_extremes",
    "C12": "forward_return",
    "C13": "hac_mean_difference",
}
RETIRED_CODES: dict[str, str] = {}


def test_registry_keys_and_functions_match_the_specs() -> None:
    for name, spec in REGISTRY.items():
        assert spec.name == name
        assert spec.func is getattr(capabilities, name)


def test_every_capability_has_a_unique_well_formed_code() -> None:
    codes = [spec.code for spec in REGISTRY.values()]
    for name, code in zip(REGISTRY, codes, strict=True):
        assert isinstance(code, str) and re.fullmatch(r"C\d{2}", code), (name, code)
    duplicates = sorted({code for code in codes if codes.count(code) > 1})
    assert duplicates == [], f"duplicate capability codes: {duplicates}"


def test_codes_are_never_renumbered_or_reused() -> None:
    current = {spec.code: name for name, spec in REGISTRY.items()}
    for code, name in ASSIGNED_CODES.items():
        assert current.get(code) == name, f"{code} was assigned to {name}"
    for code, name in RETIRED_CODES.items():
        assert code not in current, f"{code} belonged to the retired {name}"
    unrecorded = sorted(set(current) - set(ASSIGNED_CODES))
    assert unrecorded == [], f"record new codes in ASSIGNED_CODES: {unrecorded}"


def test_capability_code_looks_names_up() -> None:
    assert capability_code("rolling_volatility") == "C06"
    assert capability_code("ols_hac") is None  # retired before codes existed
    assert capability_code("arima") is None  # never registered


def test_function_signatures_match_specs() -> None:
    """Each function's keyword arguments are exactly its spec's inputs + params (+ catalog)."""
    for spec in REGISTRY.values():
        expected = [i.name for i in spec.inputs] + [p.name for p in spec.params]
        if spec.reads_catalog:
            expected.append("catalog")
        assert list(inspect.signature(spec.func).parameters) == expected, spec.name


def test_defaults_respect_their_own_constraints() -> None:
    for spec in REGISTRY.values():
        for p in spec.params:
            if p.required or p.default is None:
                continue
            if p.choices is not None:
                assert p.default in p.choices, (spec.name, p.name)
            if p.min is not None:
                assert p.min <= p.default <= p.max, (spec.name, p.name)


def test_llm_summary_is_complete_and_hides_internals() -> None:
    summary = json.loads(llm_summary())
    assert [c["name"] for c in summary] == list(REGISTRY)
    text = llm_summary()
    assert "func" not in summary[0]
    assert "library_calls" not in text and "<function" not in text
    by_name = {c["name"]: c for c in summary}
    returns = by_name["compute_returns"]
    assert returns["params"][0]["choices"] == ["simple", "log"]
    assert returns["inputs"][0]["kinds"] == ["price_level", "index_level"]
    period = by_name["filter_period"]
    assert [(p["name"], p["type"], p.get("nullable")) for p in period["params"]] == [
        ("start", "date", True),
        ("end", "date", True),
    ]


def test_registry_version_is_a_stable_hash() -> None:
    version = registry_version()
    assert len(version) == 64
    assert version == registry_version()


def test_summary_params_name_real_parameters_and_stay_out_of_the_llm_summary() -> None:
    for spec in REGISTRY.values():
        assert all(spec.param(name) for name in spec.summary_params), spec.name
    assert "summary_params" not in llm_summary()  # display only: the registry version ignores it


def test_codes_stay_out_of_the_llm_summary() -> None:
    """Display and audit only: the planner, the registry version and the plan cache ignore them."""
    summary = json.loads(llm_summary())
    assert all("code" not in capability for capability in summary)
    assert not any(spec.code in llm_summary() for spec in REGISTRY.values())
