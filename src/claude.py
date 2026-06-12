from typing import Any, Unpack

from anthropic import Anthropic
from anthropic.types import Message

from chat_client import ChatClient, ChatParams


class Claude(ChatClient[Anthropic, Message]):
    def __init__(self, client: Anthropic, model: str) -> None:
        super().__init__(client, model)

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
                text = block.text
                if block.citations:
                    titles = self._extract_citation_titles(message)
                    if titles:
                        text += " " + " ".join(f"[{t}]" for t in titles)

                parts.append(text)

        return "\n".join(parts)

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

        return response

    def chat_stream(self, messages: list[Any], **kwargs: Unpack[ChatParams]) -> Message:
        params = self._build_params(messages=messages, **kwargs)

        with self.client.messages.stream(**params) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
            print()

        final_message = stream.get_final_message()
        titles = self._extract_citation_titles(final_message)

        if titles:
            print("Sources: " + ", ".join(titles))

        return final_message
