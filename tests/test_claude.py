from typing import Any
from unittest.mock import MagicMock

import pytest

from chat_client import ChatClient
from claude import Claude

model = "test-model"


def test_cannot_instantiate_chat_client_directly() -> None:
    with pytest.raises(TypeError):
        ChatClient(client=None, model="test")  # type: ignore[abstract]


def test_claude_chat_calls_api() -> None:
    mock_client = MagicMock()
    text_block = MagicMock(type="text", text="mocked response", citations=None)
    mock_client.messages.create.return_value.content = [text_block]

    chat_client = Claude(client=mock_client, model=model)
    messages: list[Any] = [{"role": "user", "content": "hello"}]
    result = chat_client.chat(messages=messages)

    assert result.text == "mocked response"


def test_chat_excludes_system_when_none() -> None:
    mock_client = MagicMock()
    mock_client.messages.create.return_value.content = [MagicMock(text="response")]

    chat_client = Claude(client=mock_client, model=model)
    messages: list[Any] = [{"role": "user", "content": "hello"}]
    chat_client.chat(messages=messages)

    call_kwargs = mock_client.messages.create.call_args[1]
    assert "system" not in call_kwargs


def test_chat_includes_system_when_provided() -> None:
    mock_client = MagicMock()
    mock_client.messages.create.return_value.content = [MagicMock(text="response")]

    chat_client = Claude(client=mock_client, model=model)
    messages: list[Any] = [{"role": "user", "content": "hello"}]
    chat_client.chat(messages=messages, system="You are helpful")

    call_kwargs = mock_client.messages.create.call_args[1]
    assert "system" in call_kwargs


def test_build_params_web_search_enabled() -> None:
    chat_client = Claude(client=MagicMock(), model=model)
    messages: list[Any] = [{"role": "user", "content": "hello"}]
    params = chat_client._build_params(messages, web_search=True)

    assert "tools" in params
    assert params["tools"][-1]["type"] == "web_search_20250305"


def test_build_params_no_web_search_by_default() -> None:
    chat_client = Claude(client=MagicMock(), model=model)
    messages: list[Any] = [{"role": "user", "content": "hello"}]
    params = chat_client._build_params(messages)

    assert "tools" not in params


def test_build_params_thinking() -> None:
    chat_client = Claude(client=MagicMock(), model=model)
    messages: list[Any] = [{"role": "user", "content": "hello"}]
    params = chat_client._build_params(
        messages, thinking=True, thinking_budget=2048, temperature=0.5
    )

    assert params["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert params["temperature"] == 1.0


def test_build_params_caches_last_tool() -> None:
    chat_client = Claude(client=MagicMock(), model=model)
    messages: list[Any] = [{"role": "user", "content": "hello"}]
    tools = [{"name": "tool_a"}, {"name": "tool_b"}]

    params = chat_client._build_params(messages, tools=tools)

    assert params["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in params["tools"][0]
    assert "cache_control" not in tools[-1]
