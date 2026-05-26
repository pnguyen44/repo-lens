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
    mock_client.messages.create.return_value.content = [
        MagicMock(text="mocked response")
    ]

    chat_client = Claude(client=mock_client, model=model)
    messages: list[Any] = [{"role": "user", "content": "hello"}]
    result = chat_client.chat(messages=messages)

    assert result.content[0].text == "mocked response"


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
