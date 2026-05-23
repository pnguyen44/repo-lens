import pytest
from unittest.mock import MagicMock
from llm import LLM
from claude import Claude

model = "test-model"


def test_cannot_instantiate_llm_directly() -> None:
    with pytest.raises(TypeError):
        LLM(client=None, model="test")  # type: ignore[abstract]


def test_claude_chat_stores_messages() -> None:
    mock_client = MagicMock()
    mock_client.messages.create.return_value.content = [
        MagicMock(text="mocked response")
    ]

    claude = Claude(client=mock_client, model=model)
    result = claude.chat("hello")

    assert result == "mocked response"
    assert len(claude.messages) == 2
    assert claude.messages[0] == {"role": "user", "content": "hello"}
    assert claude.messages[1] == {"role": "assistant", "content": "mocked response"}


def test_chat_excludes_system_when_none() -> None:
    mock_client = MagicMock()
    mock_client.messages.create.return_value.content = [MagicMock(text="response")]

    claude = Claude(client=mock_client, model=model)
    claude.chat("hello")

    call_kwargs = mock_client.messages.create.call_args[1]
    assert "system" not in call_kwargs


def test_chat_includes_system_when_provided() -> None:
    mock_client = MagicMock()
    mock_client.messages.create.return_value.content = [MagicMock(text="response")]

    claude = Claude(client=mock_client, model=model)
    claude.chat("hello", system="You are helpful")

    call_kwargs = mock_client.messages.create.call_args[1]
    assert "system" in call_kwargs
