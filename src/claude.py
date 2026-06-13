import logging
from typing import Any, Unpack

from anthropic import Anthropic
from anthropic.types import Message

from chat_client import ChatClient, ChatParams
from token_tracker import TokenTracker

logger = logging.getLogger(__name__)


class Claude(ChatClient[Anthropic, Message]):
    def __init__(self, client: Anthropic, model: str) -> None:
        super().__init__(client, model)
        self.token_tracker = TokenTracker()

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

    def _extract_citation_titles(self, message: Message) -> set[str]:
        titles: set[str] = set()
        for block in message.content:
            if block.type == "text" and block.citations:
                for c in block.citations:
                    title = getattr(c, "document_title", None)
                    if title:
                        titles.add(title)
        return titles

    def text_from_message(self, message: Message) -> str:
        parts = []
        for block in message.content:
            if block.type == "text":
                parts.append(block.text)

        text = "\n".join(parts)
        titles = self._extract_citation_titles(message)
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

        tools = kwargs.get("tools")
        if tools:
            tools_clone = tools.copy()
            last_tool = {**tools_clone[-1]}
            last_tool["cache_control"] = {"type": "ephemeral"}
            tools_clone[-1] = last_tool
            params["tools"] = tools_clone

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

    def chat(self, messages: list[Any], **kwargs: Unpack[ChatParams]) -> Message:
        params = self._build_params(messages=messages, **kwargs)

        response = self.client.messages.create(**params)

        self.token_tracker.record(response.usage)
        logger.info(
            "Tokens: in=%d out=%d cache_read=%d",
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.usage.cache_read_input_tokens or 0,
        )

        return response

    def chat_stream(self, messages: list[Any], **kwargs: Unpack[ChatParams]) -> Message:
        params = self._build_params(messages=messages, **kwargs)

        with self.client.messages.stream(**params) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
            print()

        final_message = stream.get_final_message()

        self.token_tracker.record(final_message.usage)
        logger.info(
            "Tokens: in=%d out=%d cache_read=%d",
            final_message.usage.input_tokens,
            final_message.usage.output_tokens,
            final_message.usage.cache_read_input_tokens or 0,
        )

        titles = self._extract_citation_titles(final_message)

        if titles:
            print("Sources: " + ", ".join(titles))

        return final_message
