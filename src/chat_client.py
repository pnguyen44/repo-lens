from abc import ABC, abstractmethod
from typing import Any, Generic, TypedDict, TypeVar, Unpack


class ChatParams(TypedDict, total=False):
    system: str | None
    max_tokens: int
    temperature: float
    tools: list[Any] | None
    tool_choice: dict[str, str] | str | None
    betas: list[Any] | None
    stop_sequences: list[str] | None


# T = the LLM client type (Anthropic, OpenAI, etc.)
# R = the response type (Message, ChatCompletion, etc.)
T = TypeVar("T")
R = TypeVar("R")


class ChatClient(ABC, Generic[T, R]):
    def __init__(self, client: T, model: str) -> None:
        self.client = client
        self.model = model

    def add_user_message(self, messages: list[Any], content: str | list[Any]) -> None:
        messages.append({"role": "user", "content": content})

    def add_assistant_message(
        self, messages: list[Any], message: str | list[Any]
    ) -> None:
        messages.append(
            {
                "role": "assistant",
                "content": message.content if hasattr(message, "content") else message,
            }
        )

    @abstractmethod
    def chat(self, messages: list[Any], **kwargs: Unpack[ChatParams]) -> R:
        pass

    @abstractmethod
    def chat_stream(self, messages: list[Any], **kwargs: Unpack[ChatParams]) -> R:
        pass

    @abstractmethod
    def text_from_message(self, message: Any) -> str:
        pass
