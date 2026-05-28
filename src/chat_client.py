from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

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
    def chat(
        self,
        messages: list[Any],
        system: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 1.0,
        tools: list[Any] | None = None,
    ) -> R:
        pass

    @abstractmethod
    def text_from_message(self, message: Any) -> str:
        pass
