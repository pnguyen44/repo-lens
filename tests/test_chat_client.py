from typing import Any, Unpack

from chat_client import ChatClient, ChatParams

model = "test-model"


class FakeChatClient(ChatClient[None, str]):
    def chat(self, messages: list[Any], **kwargs: Unpack[ChatParams]) -> str:
        return "fake response"

    def chat_stream(self, messages: list[Any], **kwargs: Unpack[ChatParams]) -> str:
        return "fake streamed response"

    def text_from_message(self, message: Any) -> str:
        return str(message)


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
