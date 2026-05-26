from typing import Any

from anthropic import Anthropic
from anthropic.types import Message

from chat_client import ChatClient


class Claude(ChatClient[Anthropic, Message]):
    def __init__(self, client: Anthropic, model: str) -> None:
        super().__init__(client, model)

    def text_from_message(self, message: Message) -> str:
        return "\n".join(
            [block.text for block in message.content if block.type == "text"]
        )

    def chat(
        self,
        messages: list[Any],
        system: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 1.0,
        tools: list[Any] | None = None,
    ) -> Message:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system:
            params["system"] = system

        if tools:
            params["tools"] = tools

        response = self.client.messages.create(
            **params,
            messages=messages,
        )

        return response
