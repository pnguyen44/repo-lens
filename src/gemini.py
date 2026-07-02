import logging
from typing import Any, Unpack, Iterator

from chat_client import (
    ChatClient,
    ChatParams,
    MessageStream,
    StreamChunk,
    StreamResponse,
)
from token_tracker import UsagePayload

logger = logging.getLogger(__name__)


class GeminiStream:
    def __init__(self, stream: Any) -> None:
        self._stream: Any = stream
        self._interaction: Any = None
        self._text_parts: list[str] = []

    def __enter__(self) -> "GeminiStream":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def __iter__(self) -> Iterator[StreamChunk]:
        return self

    def __next__(self) -> StreamChunk:
        while True:
            chunk = next(self._stream)

            if chunk.event_type == "interaction.completed":
                self._interaction = chunk.interaction
                raise StopIteration

            if chunk.event_type == "step.delta" and chunk.delta.type == "text":
                self._text_parts.append(chunk.delta.text)
                return StreamChunk(type="text", text=chunk.delta.text)

    def get_final_message(self) -> StreamResponse:
        text = "".join(self._text_parts)
        if not self._interaction:
            return StreamResponse(text=text)

        usage: UsagePayload | None = None
        if self._interaction.usage:
            usage = {
                "input_tokens": self._interaction.usage.total_input_tokens,
                "output_tokens": self._interaction.usage.total_output_tokens,
            }

        return StreamResponse(
            text=text,
            stop_reason=self._interaction.status,
            usage=usage,
            raw=self._interaction,
        )


class Gemini(ChatClient[Any, Any]):
    def __init__(self, client: Any, model: str) -> None:
        super().__init__(client, model)

    def build_document_block(self, content: str, title: str) -> dict[str, Any]:
        return {
            "type": "text",
            "text": f'<source title="{title}">\n{content}\n</source>',
        }

    def add_user_message(self, messages: list[Any], content: str | list[Any]) -> None:
        if isinstance(content, str):
            text = content
        else:
            text = "\n\n".join(block["text"] for block in content)

        messages.append(
            {"type": "user_input", "content": [{"type": "text", "text": text}]}
        )

    def add_assistant_message(self, messages: list[Any], message: Any) -> None:
        if hasattr(message, "steps") and message.steps:
            for step in message.steps:
                messages.append(step.model_dump())
            return

        text = (
            message.output_text
            if hasattr(message, "output_text") and message.output_text
            else ""
        )
        messages.append(
            {"type": "model_output", "content": [{"type": "text", "text": text}]}
        )

    def chat(self, messages: list[Any], **kwargs: Unpack[ChatParams]) -> Any:
        params: dict[str, Any] = {
            "model": self.model,
            "store": False,
            "input": messages,
        }

        system = kwargs.get("system")
        if system:
            params["system_instruction"] = system

        response = self.client.interactions.create(**params)
        if getattr(response, "usage", None):
            self.record_usage(
                {
                    "input_tokens": response.usage.total_input_tokens,
                    "output_tokens": response.usage.total_output_tokens,
                }
            )
        return response

    def chat_stream(
        self, messages: list[Any], **kwargs: Unpack[ChatParams]
    ) -> MessageStream:
        params: dict[str, Any] = {
            "model": self.model,
            "input": messages,
            "stream": True,
            "store": False,
        }

        system = kwargs.get("system")

        if system:
            params["system_instruction"] = system

        stream = self.client.interactions.create(**params)

        return GeminiStream(stream)

    def text_from_message(self, message: Any) -> str:
        return message.output_text or ""

    def record_usage(self, usage: Any) -> None:
        if not usage:
            return

        self.token_tracker.record(usage)
        logger.info(
            "Tokens: in=%d out=%d",
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )
