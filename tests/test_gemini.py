from types import SimpleNamespace
from typing import Any

import pytest
from google.genai.errors import ClientError as GeminiClientError

from repo_lens.providers.chat_client import StreamError
from repo_lens.providers.gemini import GeminiStream


def _event(event_type: str, **kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(event_type=event_type, **kwargs)


def test_gemini_stream_accumulates_arguments_delta() -> None:
    events = [
        _event(
            "step.start",
            index=0,
            step=SimpleNamespace(
                type="function_call",
                id="call-1",
                name="delegate_to_agent",
                arguments=None,
            ),
        ),
        _event(
            "step.delta",
            index=0,
            delta=SimpleNamespace(
                type="arguments_delta",
                arguments='{"agent_name": "rag", "task": "auth"}',
                partial_arguments=None,
            ),
        ),
        _event("step.stop", index=0),
        _event(
            "interaction.completed",
            interaction=SimpleNamespace(status="requires_action", usage=None),
        ),
    ]

    stream = GeminiStream(iter(events))
    list(stream)
    response = stream.get_final_message()

    assert response.stop_reason == "tool_use"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].input == {
        "agent_name": "rag",
        "task": "auth",
    }


def test_gemini_stream_accumulates_legacy_arguments() -> None:
    events = [
        _event(
            "step.start",
            index=1,
            step=SimpleNamespace(
                type="function_call",
                id="call-2",
                name="delegate_to_agent",
                arguments=None,
            ),
        ),
        _event(
            "step.delta",
            index=1,
            delta=SimpleNamespace(
                type="arguments",
                partial_arguments='{"agent_name": "github", "task": "list PRs"}',
                arguments=None,
            ),
        ),
        _event("step.stop", index=1),
        _event(
            "interaction.completed",
            interaction=SimpleNamespace(status="requires_action", usage=None),
        ),
    ]

    stream = GeminiStream(iter(events))
    list(stream)
    response = stream.get_final_message()

    assert response.tool_calls[0].input == {
        "agent_name": "github",
        "task": "list PRs",
    }


def test_gemini_stream_raises_client_error_on_rate_limit() -> None:
    events = [
        _event(
            "error",
            error=SimpleNamespace(
                message="You exceeded your current quota. Please retry in 3.9s."
            ),
        ),
    ]

    stream = GeminiStream(iter(events))
    with pytest.raises(GeminiClientError) as exc_info:
        list(stream)

    assert exc_info.value.code == 429


def test_gemini_stream_raises_stream_error_on_non_rate_limit() -> None:
    events = [
        _event("error", error=SimpleNamespace(message="Internal server error")),
    ]

    stream = GeminiStream(iter(events))
    with pytest.raises(StreamError):
        list(stream)
