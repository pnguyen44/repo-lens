import logging
from typing import Any, Unpack, Iterator

from google.genai import Client as GenaiClient

from chat_client import (
    ChatClient,
    ChatParams,
    MessageStream,
    StreamChunk,
    StreamResponse,
)
from token_tracker import UsagePayload


logger = logging.getLogger(__name__)


def _extract_usage(response: Any) -> UsagePayload | None:
    if not getattr(response, "usage", None):
        return None

    return {
        "input_tokens": response.usage.total_input_tokens,
        "output_tokens": response.usage.total_output_tokens,
    }


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

        usage = _extract_usage(self._interaction)

        return StreamResponse(
            text=text,
            stop_reason=self._interaction.status,
            usage=usage,
            raw=self._interaction,
        )


class Gemini(ChatClient[GenaiClient]):
    def __init__(self, client: GenaiClient, model: str) -> None:
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

    def add_assistant_message(
        self, messages: list[Any], message: StreamResponse | str
    ) -> None:
        if isinstance(message, str):
            messages.append(
                {"type": "model_output", "content": [{"type": "text", "text": message}]}
            )
            return

        raw = message.raw
        if hasattr(raw, "steps") and raw.steps:
            for step in raw.steps:
                messages.append(step.model_dump())
            return

        messages.append(
            {
                "type": "model_output",
                "content": [{"type": "text", "text": message.text}],
            }
        )

    def chat(self, messages: list[Any], **kwargs: Unpack[ChatParams]) -> StreamResponse:
        params: dict[str, Any] = {
            "model": self.model,
            "store": False,
            "input": messages,
        }

        system = kwargs.get("system")
        if system:
            params["system_instruction"] = system

        response = self.client.interactions.create(**params)

        usage = _extract_usage(response)

        if usage:
            self.record_usage(usage)

        return StreamResponse(
            text=self._text_from_message(response),
            stop_reason=getattr(response, "status", ""),
            usage=usage,
            raw=response,
        )

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

    def _text_from_message(self, message: Any) -> str:
        return message.output_text or ""

    def record_usage(self, usage: UsagePayload) -> None:
        if not usage:
            return

        self.token_tracker.record(usage)
        logger.info(
            "Tokens: in=%d out=%d",
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )
