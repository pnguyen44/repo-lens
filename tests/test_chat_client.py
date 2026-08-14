from collections.abc import AsyncIterator
from typing import Any, Unpack

from repo_lens.providers.chat_client import (
    ChatClient,
    ChatParams,
    ChatResponse,
    MessageStream,
    StreamChunk,
)

model = "test-model"


class FakeStream:
    async def __aenter__(self) -> "FakeStream":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    def __aiter__(self) -> AsyncIterator[StreamChunk]:
        return self._iter_chunks()

    async def _iter_chunks(self) -> AsyncIterator[StreamChunk]:
        if False:
            yield StreamChunk(type="text")

    async def get_final_message(self) -> ChatResponse:
        return ChatResponse(text="fake message")


class FakeChatClient(ChatClient[None]):
    def chat(self, messages: list[Any], **kwargs: Unpack[ChatParams]) -> ChatResponse:
        return ChatResponse(text="fake response")

    async def chat_stream(
        self, messages: list[Any], **kwargs: Unpack[ChatParams]
    ) -> MessageStream:
        return FakeStream()

    def record_usage(self, usage: Any) -> None:
        pass


def test_add_user_message() -> None:
    fake_client = FakeChatClient(client=None, model=model)
    messages: list[Any] = []

    fake_client.add_user_message(messages, "What is 1 + 1?")

    assert len(messages) == 1
    assert messages[0] == {"role": "user", "content": "What is 1 + 1?"}


def test_add_assistant_message() -> None:
    fake_client = FakeChatClient(client=None, model=model)
    messages: list[Any] = []

    fake_client.add_user_message(messages, "What is 1 + 1?")
    fake_client.add_assistant_message(messages, "1 + 1 = 2")

    assert len(messages) == 2
    assert messages[1] == {"role": "assistant", "content": "1 + 1 = 2"}
