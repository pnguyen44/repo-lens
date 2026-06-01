from typing import Any, Unpack

from anthropic import Anthropic
from anthropic.types import Message

from chat_client import ChatClient, ChatParams


class Claude(ChatClient[Anthropic, Message]):
    def __init__(self, client: Anthropic, model: str) -> None:
        super().__init__(client, model)

    def text_from_message(self, message: Message) -> str:
        return "\n".join(
            [block.text for block in message.content if block.type == "text"]
        )

    def _build_params(
        self, messages: list[Any], **kwargs: Unpack[ChatParams]
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 1.0),
        }

        if kwargs.get("system"):
            params["system"] = kwargs["system"]

        if kwargs.get("tools"):
            params["tools"] = kwargs["tools"]

        if kwargs.get("tool_choice"):
            params["tool_choice"] = kwargs["tool_choice"]

        if kwargs.get("betas"):
            params["betas"] = kwargs["betas"]

        if kwargs.get("stop_sequences"):
            params["stop_sequences"] = kwargs["stop_sequences"]

        return params

    def chat(self, messages: list[Any], **kwargs: Unpack[ChatParams]) -> Message:
        params = self._build_params(messages=messages, **kwargs)

        response = self.client.messages.create(**params)

        return response

    def chat_stream(self, messages: list[Any], **kwargs: Unpack[ChatParams]) -> Message:
        params = self._build_params(messages=messages, **kwargs)

        with self.client.messages.stream(**params) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
            print()

        return stream.get_final_message()
