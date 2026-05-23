from abc import ABC, abstractmethod
from typing import Any


class LLM(ABC):
    def __init__(self, client: Any, model: str) -> None:
        self.client = client
        self.model = model
        self.messages: list[dict[str, str]] = []

    def _add_message(self, role: str, text: str) -> None:
        message = {"role": role, "content": text}
        self.messages.append(message)

    def add_user_message(self, text: str) -> None:
        self._add_message("user", text)

    def add_assistant_message(self, text: str) -> None:
        self._add_message("assistant", text)

    def get_chat_history(self) -> list[dict[str, str]]:
        return self.messages

    @abstractmethod
    def chat(
        self,
        user_message: str,
        system: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 1.0,
    ) -> str:
        pass
