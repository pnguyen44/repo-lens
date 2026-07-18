import logging
from typing import Any, Iterator, Unpack

from anthropic import Anthropic
from anthropic.types import Message, Usage

from repo_lens.providers.chat_client import (
    ChatClient,
    ChatParams,
    ChatResponse,
    MessageStream,
    StreamChunk,
    ToolCall,
)
from repo_lens.providers.token_tracker import TokenTracker, UsagePayload

logger = logging.getLogger(__name__)

WEB_SEARCH_MAX_USES = 5


class ClaudeStream:
    def __init__(self, manager: Any) -> None:
        self._manager = manager
        self._message_stream: Any = None
        self._response: Any = None
        self._text_parts: list[str] = []
        self._in_tool_block = False

    def __enter__(self) -> "ClaudeStream":
        self._message_stream = self._manager.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        self._manager.__exit__(*args)

    def __iter__(self) -> Iterator[StreamChunk]:
        return self

    def __next__(self) -> StreamChunk:
        while True:
            chunk = next(self._message_stream)

            if chunk.type == "text":
                self._text_parts.append(chunk.text)
                return StreamChunk(type="text", text=chunk.text)

            if chunk.type == "content_block_start":
                if chunk.content_block.type == "tool_use":
                    self._in_tool_block = True
                    return StreamChunk(
                        type="tool_start", tool_name=chunk.content_block.name
                    )

            if chunk.type == "input_json" and chunk.partial_json:
                return StreamChunk(type="tool_input", partial_json=chunk.partial_json)

            if chunk.type == "content_block_stop" and self._in_tool_block:
                self._in_tool_block = False
                return StreamChunk(type="tool_stop")

    def get_final_message(self) -> ChatResponse:
        message = self._message_stream.get_final_message()
        self._response = message

        stop_reason = message.stop_reason or "end_turn"
        tool_calls = [
            ToolCall(id=b.id, name=b.name, input=b.input)
            for b in message.content
            if b.type == "tool_use"
        ]

        return ChatResponse(
            text="".join(self._text_parts),
            stop_reason=stop_reason,
            tool_calls=tool_calls,
            usage=message.usage,
            raw=message,
        )


class Claude(ChatClient[Anthropic]):
    def __init__(
        self, client: Anthropic, model: str, token_tracker: TokenTracker | None = None
    ) -> None:
        super().__init__(client=client, model=model, token_tracker=token_tracker)

    def build_document_block(self, content: str, title: str) -> dict[str, Any]:
        return {
            "type": "document",
            "source": {
                "type": "text",
                "media_type": "text/plain",
                "data": content,
            },
            "title": title,
            "citations": {"enabled": True},
        }

    def has_web_search_results(self, raw: Any) -> bool:
        if not raw:
            return False
        return any(b.type == "web_search_tool_result" for b in raw.content)

    def extract_citation_titles(self, message: Message) -> set[str]:
        titles: set[str] = set()
        for block in message.content:
            if block.type == "text" and block.citations:
                for c in block.citations:
                    title = getattr(c, "document_title", None)
                    if title:
                        titles.add(title)
        return titles

    def _text_from_message(self, message: Message) -> str:
        parts = []
        for block in message.content:
            if block.type == "text":
                parts.append(block.text)

        text = "\n".join(parts)
        titles = self.extract_citation_titles(message)
        if titles:
            text += "\n" + " ".join(f"[{t}]" for t in titles)
        return text

    def _build_params(
        self, messages: list[Any], **kwargs: Unpack[ChatParams]
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 1.0),
        }

        system = kwargs.get("system")
        if system:
            params["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]

        thinking = kwargs.get("thinking", False)
        thinking_budget = kwargs.get("thinking_budget", 1024)

        if thinking:
            params["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
            params["temperature"] = 1.0

        tools = kwargs.get("tools", [])
        tools_list: list[Any] = []
        if tools:
            tools_clone = tools.copy()
            last_tool = {**tools_clone[-1]}
            last_tool["cache_control"] = {"type": "ephemeral"}
            tools_clone[-1] = last_tool
            tools_list.extend(tools_clone)

        if kwargs.get("web_search"):
            tools_list.append(
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": WEB_SEARCH_MAX_USES,
                }
            )

        if tools_list:
            params["tools"] = tools_list

        tool_choice = kwargs.get("tool_choice")
        if tool_choice:
            params["tool_choice"] = tool_choice

        betas = kwargs.get("betas")
        if betas:
            params["betas"] = betas

        stop_sequences = kwargs.get("stop_sequences")
        if stop_sequences:
            params["stop_sequences"] = stop_sequences

        return params

    def record_usage(self, usage: Usage) -> None:
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        cache_read = usage.cache_read_input_tokens or 0
        cache_creation = usage.cache_creation_input_tokens or 0

        usage_payload: UsagePayload = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_creation,
        }

        self.token_tracker.record(usage_payload)
        logger.info(
            "Tokens: in=%d out=%d cache_read=%d",
            input_tokens,
            output_tokens,
            cache_read,
        )

    def chat_json(
        self, messages: list[Any], **kwargs: Unpack[ChatParams]
    ) -> ChatResponse:
        self.add_assistant_message(messages=messages, message="```json")
        kwargs["stop_sequences"] = ["```"]
        return self.chat(messages=messages, **kwargs)

    def chat(self, messages: list[Any], **kwargs: Unpack[ChatParams]) -> ChatResponse:
        params = self._build_params(messages=messages, **kwargs)

        response = self.client.messages.create(**params)

        self.record_usage(response.usage)

        return ChatResponse(
            text=self._text_from_message(response),
            stop_reason=response.stop_reason or "end_turn",
            tool_calls=[b for b in response.content if b.type == "tool_use"],
            usage=response.usage,
            raw=response,
        )

    def chat_stream(
        self, messages: list[Any], **kwargs: Unpack[ChatParams]
    ) -> MessageStream:
        params = self._build_params(messages=messages, **kwargs)
        stream = self.client.messages.stream(**params)
        return ClaudeStream(stream)
