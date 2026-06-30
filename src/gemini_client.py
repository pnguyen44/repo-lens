import logging
from typing import Any, Unpack
from chat_client import ChatClient, ChatParams, MessageStream
from token_tracker import TokenTracker

logger = logging.getLogger(__name__)


class Gemini(ChatClient[Any, Any]):
    def __init__(self, client: Any, model: str) -> None:
        super().__init__(client, model)
        self.token_tracker = TokenTracker()

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
        for step in message.steps:
            messages.append(step.model_dump())

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
        return response

    def chat_stream(
        self, messages: list[Any], **kwargs: Unpack[ChatParams]
    ) -> MessageStream:
        raise NotImplementedError

    def text_from_message(self, message: Any) -> str:
        return message.output_text or ""

    def record_usage(self, usage: Any) -> None:
        if usage:
            self.token_tracker.record(usage)
            logger.info(
                "Tokens: in=%d out=%d",
                usage.prompt_token_count or 0,
                usage.candidates_token_count or 0,
            )
