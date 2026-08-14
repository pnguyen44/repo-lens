import json
import logging
from collections.abc import AsyncIterator
from typing import Any, Unpack

from google.genai import Client as GenaiClient
from google.genai.errors import ClientError as GeminiClientError

from repo_lens.providers.chat_client import (
    ChatClient,
    ChatParams,
    ChatResponse,
    MessageStream,
    StreamChunk,
    StreamError,
    ToolCall,
)
from repo_lens.providers.token_tracker import TokenTracker, UsagePayload

logger = logging.getLogger(__name__)


def _extract_usage(response: Any) -> UsagePayload | None:
    if not getattr(response, "usage", None):
        return None

    return {
        "input_tokens": response.usage.total_input_tokens,
        "output_tokens": response.usage.total_output_tokens,
    }


def _extract_tool_calls(interaction: Any) -> list[ToolCall]:
    tool_calls = []
    for step in getattr(interaction, "steps", []):
        if getattr(step, "type", None) == "function_call":
            tool_calls.append(
                ToolCall(
                    id=step.id,
                    name=step.name,
                    input=dict(step.arguments),
                )
            )
    return tool_calls


def _is_rate_limit_message(message: str) -> bool:
    lower = message.lower()
    return "quota" in lower or "rate" in lower


class GeminiStream:
    def __init__(self, stream: Any) -> None:
        self._stream: Any = stream
        self._interaction: Any = None
        self._text_parts: list[str] = []
        self._pending_calls: dict[int, dict[str, Any]] = {}

    async def __aenter__(self) -> "GeminiStream":
        return self

    async def __aexit__(self, *args: Any) -> None:
        close = getattr(self._stream, "close", None)
        if close is not None:
            await close()

    def __aiter__(self) -> AsyncIterator[StreamChunk]:
        return self._iter_chunks()

    async def _iter_chunks(self) -> AsyncIterator[StreamChunk]:
        async for chunk in self._stream:
            match chunk.event_type:
                case "interaction.completed":
                    self._interaction = chunk.interaction
                    return

                case "error":
                    error = getattr(chunk, "error", None)
                    msg = getattr(error, "message", str(error)) if error else str(chunk)
                    logger.error("Gemini stream error: %s", msg)
                    if _is_rate_limit_message(msg):
                        raise GeminiClientError(429, {"error": {"message": msg}})
                    raise StreamError(f"Gemini stream error: {msg}")

                case "step.start" if (
                    getattr(chunk.step, "type", None) == "function_call"
                ):
                    initial_args = ""
                    if hasattr(chunk.step, "arguments") and chunk.step.arguments:
                        if isinstance(chunk.step.arguments, dict):
                            initial_args = json.dumps(chunk.step.arguments)
                        else:
                            initial_args = chunk.step.arguments

                    self._pending_calls[chunk.index] = {
                        "id": chunk.step.id,
                        "name": chunk.step.name,
                        "arguments": initial_args,
                    }

                    yield StreamChunk(type="tool_start", tool_name=chunk.step.name)

                case "step.delta":
                    match chunk.delta.type:
                        case "text":
                            self._text_parts.append(chunk.delta.text)
                            yield StreamChunk(type="text", text=chunk.delta.text)

                        case "arguments" | "arguments_delta":
                            partial = (
                                getattr(chunk.delta, "partial_arguments", None)
                                or getattr(chunk.delta, "arguments", None)
                                or ""
                            )
                            if chunk.index in self._pending_calls:
                                self._pending_calls[chunk.index]["arguments"] += partial
                            yield StreamChunk(
                                type="tool_input",
                                partial_json=partial,
                            )

                case "step.stop":
                    yield StreamChunk(type="tool_stop")

    async def get_final_message(self) -> ChatResponse:
        text = "".join(self._text_parts)

        steps: list[dict[str, Any]] = []
        tool_calls = []

        if text:
            steps.append(
                {"type": "model_output", "content": [{"type": "text", "text": text}]}
            )

        for call in self._pending_calls.values():
            call_id = call["id"]
            name = call["name"]
            args = json.loads(call["arguments"]) if call["arguments"] else {}
            tool_calls.append(ToolCall(id=call_id, name=name, input=args))
            steps.append(
                {
                    "type": "function_call",
                    "name": name,
                    "id": call_id,
                    "arguments": args,
                }
            )

        if not self._interaction:
            return ChatResponse(text=text)

        usage = _extract_usage(self._interaction)

        return ChatResponse(
            text=text,
            stop_reason="tool_use" if tool_calls else self._interaction.status,
            tool_calls=tool_calls,
            steps=steps,
            usage=usage,
            raw=self._interaction,
        )


class Gemini(ChatClient[GenaiClient]):
    def __init__(
        self, client: GenaiClient, model: str, token_tracker: TokenTracker | None = None
    ) -> None:
        super().__init__(client=client, model=model, token_tracker=token_tracker)
        self._previous_interaction_id: str | None = None

    def build_document_block(self, content: str, title: str) -> dict[str, Any]:
        return {
            "type": "text",
            "text": f'<source title="{title}">\n{content}\n</source>',
        }

    def add_user_message(self, messages: list[Any], content: str | list[Any]) -> None:
        if isinstance(content, str):
            text = content
        elif (
            content
            and isinstance(content[0], dict)
            and content[0].get("type") == "tool_result"
        ):
            for result in content:
                messages.append(
                    {
                        "type": "function_result",
                        "name": result["name"],
                        "call_id": result["tool_use_id"],
                        "result": result["content"],
                    }
                )
            return
        else:
            text = "\n\n".join(block["text"] for block in content)

        messages.append(
            {"type": "user_input", "content": [{"type": "text", "text": text}]}
        )

    def add_assistant_message(
        self, messages: list[Any], message: ChatResponse | str
    ) -> None:
        if isinstance(message, str):
            messages.append(
                {"type": "model_output", "content": [{"type": "text", "text": message}]}
            )
            return

        self._previous_interaction_id = message.raw.id
        messages.clear()

    def _build_params(
        self, messages: list[Any], **kwargs: Unpack[ChatParams]
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "input": messages,
        }

        if self._previous_interaction_id:
            params["previous_interaction_id"] = self._previous_interaction_id

        system = kwargs.get("system")
        if system:
            params["system_instruction"] = system

        tools = kwargs.get("tools")
        if tools:
            params["tools"] = self._to_gemini_tools(tools)

        return params

    def chat(self, messages: list[Any], **kwargs: Unpack[ChatParams]) -> ChatResponse:
        params = self._build_params(messages, **kwargs)

        response = self.client.interactions.create(**params)
        usage = _extract_usage(response)
        tool_calls = _extract_tool_calls(response)

        if usage:
            self.record_usage(usage)

        return ChatResponse(
            text=self._text_from_message(response),
            stop_reason="tool_use" if tool_calls else getattr(response, "status", ""),
            tool_calls=tool_calls,
            usage=usage,
            raw=response,
        )

    async def chat_stream(
        self, messages: list[Any], **kwargs: Unpack[ChatParams]
    ) -> MessageStream:
        params = self._build_params(messages, **kwargs)
        params["stream"] = True

        stream = await self.client.aio.interactions.create(**params)

        return GeminiStream(stream)

    def _text_from_message(self, message: Any) -> str:
        return message.output_text or ""

    def record_usage(self, usage: UsagePayload) -> None:
        if not usage:
            return

        self.token_tracker.record(usage)
        logger.debug(
            "Tokens: in=%d out=%d",
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )

    @staticmethod
    def _to_gemini_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            }
            for tool in tools
        ]
