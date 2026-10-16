"""The registry is internally consistent and its LLM summary is generated, not hand-written."""

import inspect
import json

from quantlab import capabilities
from quantlab.registry import REGISTRY, llm_summary, registry_version

EXPECTED = [
    "load_series",
    "resample",
    "compute_returns",
    "difference",
    "spread",
    "rolling_volatility",
    "flag_regime",
    "summary_stats",
]


def test_exactly_the_eight_capabilities() -> None:
    assert list(REGISTRY) == EXPECTED
    for name, spec in REGISTRY.items():
        assert spec.name == name
        assert spec.func is getattr(capabilities, name)


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
    assert [c["name"] for c in summary] == EXPECTED
    text = llm_summary()
    assert "func" not in summary[0]
    assert "library_calls" not in text and "<function" not in text
    returns = summary[EXPECTED.index("compute_returns")]
    assert returns["params"][0]["choices"] == ["simple", "log"]
    assert returns["inputs"][0]["kinds"] == ["price_level", "index_level"]


def test_registry_version_is_a_stable_hash() -> None:
    version = registry_version()
    assert len(version) == 64
    assert version == registry_version()
