"""Planner: mock scenarios, LLM call shape (via a fake client), error handling, plan cache."""

import copy

import anthropic
import httpx2
import pytest

from quantlab.config import Settings
from quantlab.data import Catalog, load_dataset
from quantlab.plan import plan_tool_schema
from quantlab.planner import (
    MockPlanner,
    build_system_prompt,
    cache_key,
    describe_api_error,
    generate_plan,
    plan_from_output,
)
from quantlab.scenarios import SCENARIOS, SCENARIOS_BY_ID

from fakes import FakeClient, text_response, tool_response

PLAN_A = SCENARIOS_BY_ID["A"].plan


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
def test_mock_planner_returns_each_canned_plan(scenario) -> None:
    result = MockPlanner().generate_plan(scenario.request, list(scenario.dataset_ids))
    assert result.plan is not None and result.issues == []
    assert result.plan.request == scenario.request
    assert [s.capability for s in result.plan.steps] == [
        s["capability"] for s in scenario.plan["steps"]
    ]


def test_mock_planner_refuses_unknown_requests() -> None:
    result = MockPlanner().generate_plan("Price a Bermudan swaption", ["DGS10"])
    assert result.plan is not None and result.plan.cannot_fulfill
    assert "live" in result.plan.refusal_reason


def test_plan_from_output_sets_system_fields_and_rejects_junk() -> None:
    plan, issues = plan_from_output({**PLAN_A, "request": "tampered"}, "the real request")
    assert issues == [] and plan.request == "the real request"
    assert plan.plan_id.startswith("plan-")
    assert plan_from_output("not a dict", "r")[0] is None
    bad = copy.deepcopy(PLAN_A)
    bad["steps"][0]["code"] = "import os"
    plan, issues = plan_from_output(bad, "r")
    assert plan is None and issues[0].code == "G01"


def test_system_prompt_has_registry_and_metadata_but_no_values(catalog: Catalog) -> None:
    prompt = build_system_prompt(catalog, ["NASDAQCOM"])
    assert '"name": "flag_regime"' in prompt and "cannot_fulfill" in prompt
    assert "NASDAQCOM" in prompt and "DGS10" not in prompt
    # No data value may leak into the prompt, e.g. the latest price.
    last_price = load_dataset(catalog, "NASDAQCOM").series.iloc[-1]
    assert f"{last_price:.4f}" not in prompt


def test_live_call_shape_and_success(settings: Settings, catalog: Catalog) -> None:
    client = FakeClient(tool_response(copy.deepcopy(PLAN_A)))
    result = generate_plan("weekly nasdaq volatility", ["NASDAQCOM"], "live", settings, catalog,
                           client)  # fmt: skip
    assert result.plan is not None and result.error is None
    assert (result.provider, result.mode, result.from_cache) == ("anthropic", "live", False)
    assert result.input_tokens == 1200 and result.served_model == "claude-sonnet-5-5"
    call = client.calls[0]
    assert call["model"] == settings.planner_model
    assert call["tools"][0]["input_schema"] == plan_tool_schema()
    assert call["tool_choice"] == {"type": "auto"}  # forced tool_choice is rejected by the API
    assert "temperature" not in call  # not accepted by current models
    assert "<user_request>" in call["messages"][0]["content"]


@pytest.mark.parametrize(
    ("client", "expected"),
    [
        (FakeClient(text_response("Sure! Here is my plan: ...")), "instead of calling"),
        (FakeClient(text_response("", stop_reason="refusal")), "declined"),
        (FakeClient(text_response("...", stop_reason="max_tokens")), "cut off"),
        (FakeClient(tool_response({"steps": "nonsense"})), "does not match"),
        (FakeClient(error=RuntimeError("socket closed")), "socket closed"),
    ],
)
def test_planner_failures_become_g01(
    client: FakeClient, expected: str, settings: Settings, catalog: Catalog
) -> None:
    result = generate_plan("anything", ["NASDAQCOM"], "live", settings, catalog, client)
    assert result.plan is None
    assert expected in result.error
    assert {issue.code for issue in result.issues} == {"G01"}


def test_describe_api_error_is_specific() -> None:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    assert "network" in describe_api_error(anthropic.APIConnectionError(request=request))
    assert "timed out" in describe_api_error(anthropic.APITimeoutError(request=request))


def test_cached_mode_reuses_the_plan(settings: Settings, catalog: Catalog) -> None:
    client = FakeClient(tool_response(copy.deepcopy(PLAN_A)))
    first = generate_plan("req", ["NASDAQCOM"], "cached", settings, catalog, client)
    second = generate_plan("req", ["NASDAQCOM"], "cached", settings, catalog, client)
    assert not first.from_cache and second.from_cache
    assert len(client.calls) == 1  # the second call never reached the API
    assert second.plan.steps == first.plan.steps
    assert second.plan.plan_id != first.plan.plan_id  # a new plan instance each time


def test_failures_are_not_cached(settings: Settings, catalog: Catalog) -> None:
    failing = FakeClient(error=RuntimeError("down"))
    generate_plan("req", ["NASDAQCOM"], "cached", settings, catalog, failing)
    assert not settings.cache_dir.exists() or not any(settings.cache_dir.iterdir())


def test_cache_key_depends_on_inputs() -> None:
    base = cache_key("req", ["A", "B"], "m")
    assert base == cache_key("req", ["B", "A"], "m")  # dataset order does not matter
    assert base != cache_key("req2", ["A", "B"], "m")
    assert base != cache_key("req", ["A"], "m")
    assert base != cache_key("req", ["A", "B"], "other-model")
