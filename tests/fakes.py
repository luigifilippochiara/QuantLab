"""A tiny stand-in for the Anthropic client, so planner and narrator tests need no network.

It mimics only what QuantLab uses: client.beta.messages.create(**kwargs) returning
an object with .content (blocks with .type/.name/.input/.text), .stop_reason,
.model and .usage. Every call's keyword arguments are recorded for assertions.
"""

from dataclasses import dataclass, field
from typing import Any


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


class FakeMessages:
    """Records calls and returns a fixed response (or raises a fixed exception)."""

    def __init__(self, response: FakeResponse | None, error: Exception | None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeBeta:
    """Holds .messages, like client.beta."""

    def __init__(self, messages: FakeMessages) -> None:
        self.messages = messages


class FakeClient:
    """client.beta.messages.create(...) -> canned response."""

    def __init__(self, response: FakeResponse | None = None, error: Exception | None = None):
        self.beta = FakeBeta(FakeMessages(response, error))

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.beta.messages.calls


def tool_response(plan_input: Any) -> FakeResponse:
    """A response in which the model called submit_plan with `plan_input`."""
    return FakeResponse(content=[FakeBlock(type="tool_use", name="submit_plan", input=plan_input)])


def text_response(text: str, stop_reason: str = "end_turn") -> FakeResponse:
    """A response containing only text."""
    return FakeResponse(content=[FakeBlock(type="text", text=text)], stop_reason=stop_reason)
