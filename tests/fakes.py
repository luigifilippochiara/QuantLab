"""Stand-ins for the OpenAI and Anthropic clients, so LLM tests need no network or key.

They mimic only what QuantLab uses, and record every call's keyword arguments:
- OpenAI:    client.responses.create(**kwargs) -> .output (items with .type/.name/
             .arguments), .output_text, .status, .incomplete_details, .model, .usage
- Anthropic: client.beta.messages.create(**kwargs) -> .content (blocks with
             .type/.name/.input/.text), .stop_reason, .model, .usage
"""

import json
from dataclasses import dataclass, field
from typing import Any


class FakeEndpoint:
    """Records calls to .create() and returns a fixed response (or raises a fixed error).

    Given a list, it returns its responses in order, one per call (e.g. a plan, then a
    revised plan, then a narrative), and fails on a call beyond the last: a test then
    knows exactly how many LLM calls were made.
    """

    def __init__(self, response: Any, error: Exception | None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if isinstance(self.response, list):
            if len(self.calls) > len(self.response):
                raise AssertionError(f"unexpected LLM call #{len(self.calls)}: none was canned")
            return self.response[len(self.calls) - 1]
        return self.response


# ---------------------------------------------------------------------------
# OpenAI (Responses API)
# ---------------------------------------------------------------------------


@dataclass
class FakeOutputItem:
    """An item of response.output (a function call or a message)."""

    type: str
    name: str = ""
    arguments: str = ""


@dataclass
class FakeIncompleteDetails:
    """Why a response is incomplete."""

    reason: str


@dataclass
class FakeOpenAIUsage:
    """Token usage of a response."""

    input_tokens: int = 1500
    output_tokens: int = 400


@dataclass
class FakeOpenAIResponse:
    """A Responses API response."""

    output: list[FakeOutputItem]
    output_text: str = ""
    status: str = "completed"
    incomplete_details: FakeIncompleteDetails | None = None
    model: str = "gpt-6-luna"
    usage: FakeOpenAIUsage = field(default_factory=FakeOpenAIUsage)


class FakeOpenAIClient:
    """client.responses.create(...) -> canned response."""

    def __init__(
        self,
        response: FakeOpenAIResponse | list[FakeOpenAIResponse] | None = None,
        error: Exception | None = None,
    ):
        self.responses = FakeEndpoint(response, error)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.responses.calls


def openai_tool_response(plan_input: Any) -> FakeOpenAIResponse:
    """A response in which the model called submit_plan with `plan_input` (sent as JSON text)."""
    arguments = plan_input if isinstance(plan_input, str) else json.dumps(plan_input)
    return FakeOpenAIResponse(
        output=[FakeOutputItem(type="function_call", name="submit_plan", arguments=arguments)]
    )


def openai_text_response(
    text: str, status: str = "completed", reason: str = ""
) -> FakeOpenAIResponse:
    """A response containing only text (optionally incomplete, with a reason)."""
    details = FakeIncompleteDetails(reason) if reason else None
    return FakeOpenAIResponse(
        output=[FakeOutputItem(type="message")],
        output_text=text,
        status=status,
        incomplete_details=details,
    )


# ---------------------------------------------------------------------------
# Anthropic (Messages API)
# ---------------------------------------------------------------------------


@dataclass
class FakeBlock:
    """A response content block."""

    type: str
    name: str = ""
    input: Any = None
    text: str = ""


@dataclass
class FakeUsage:
    """Token usage of a response."""

    input_tokens: int = 1200
    output_tokens: int = 300


@dataclass
class FakeResponse:
    """A Messages API response."""

    content: list[FakeBlock]
    stop_reason: str = "tool_use"
    model: str = "claude-sonnet-5-5"
    usage: FakeUsage = field(default_factory=FakeUsage)


class FakeBeta:
    """Holds .messages, like client.beta."""

    def __init__(self, messages: FakeEndpoint) -> None:
        self.messages = messages


class FakeAnthropicClient:
    """client.beta.messages.create(...) -> canned response."""

    def __init__(
        self,
        response: FakeResponse | list[FakeResponse] | None = None,
        error: Exception | None = None,
    ):
        self.beta = FakeBeta(FakeEndpoint(response, error))

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.beta.messages.calls


def tool_response(plan_input: Any) -> FakeResponse:
    """A response in which the model called submit_plan with `plan_input`."""
    return FakeResponse(content=[FakeBlock(type="tool_use", name="submit_plan", input=plan_input)])


def text_response(text: str, stop_reason: str = "end_turn") -> FakeResponse:
    """A response containing only text."""
    return FakeResponse(content=[FakeBlock(type="text", text=text)], stop_reason=stop_reason)
