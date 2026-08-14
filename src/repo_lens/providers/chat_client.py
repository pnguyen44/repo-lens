import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, Self, TypedDict, TypeVar, Unpack

from repo_lens.providers.token_tracker import TokenTracker


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class StreamChunk:
    type: str
    text: str = ""
    tool_name: str = ""
    partial_json: str = ""


@dataclass
class ChatResponse:
    text: str = ""
    stop_reason: str = ""
    tool_calls: list[Any] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    usage: Any = None
    raw: Any = None


class StreamError(Exception):
    pass


class MessageStream(Protocol):
    async def __aenter__(self) -> Self: ...
    async def __aexit__(self, *args: Any) -> None: ...
    def __aiter__(self) -> AsyncIterator[StreamChunk]: ...
    async def get_final_message(self) -> ChatResponse: ...


class ChatParams(TypedDict, total=False):
    system: str | None
    max_tokens: int
    temperature: float
    tools: list[Any] | None
    tool_choice: dict[str, Any] | str | None
    betas: list[Any] | None
    stop_sequences: list[str] | None
    thinking: bool | None
    thinking_budget: int | None
    web_search: bool | None


# T = the LLM client type (Anthropic, OpenAI, etc.)
T = TypeVar("T")


class ChatClientProtocol(Protocol):
    model: str
    token_tracker: TokenTracker

    def add_user_message(
        self, messages: list[Any], content: str | list[Any]
    ) -> None: ...

    def add_assistant_message(
        self, messages: list[Any], message: ChatResponse | str
    ) -> None: ...

    def build_document_block(self, content: str, title: str) -> dict[str, Any]: ...

    def chat_json(
        self, messages: list[Any], **kwargs: Unpack[ChatParams]
    ) -> ChatResponse: ...

    def chat(
        self, messages: list[Any], **kwargs: Unpack[ChatParams]
    ) -> ChatResponse: ...

    async def chat_stream(
        self, messages: list[Any], **kwargs: Unpack[ChatParams]
    ) -> MessageStream: ...

    def record_usage(self, usage: Any) -> None: ...

    def extract_citation_titles(self, message: Any) -> set[str]: ...

    def has_web_search_results(self, raw: Any) -> bool: ...


class ChatClient(ABC, Generic[T]):
    def __init__(
        self, client: T, model: str, token_tracker: TokenTracker | None = None
    ) -> None:
        self.client = client
        self.model = model
        self.token_tracker = token_tracker or TokenTracker()

    def add_user_message(self, messages: list[Any], content: str | list[Any]) -> None:
        messages.append({"role": "user", "content": content})

    def add_assistant_message(
        self, messages: list[Any], message: ChatResponse | str
    ) -> None:
        if isinstance(message, str):
            content: Any = message
        else:
            if message.raw and hasattr(message.raw, "content"):
                content = message.raw.content
            else:
                content = message.text
        messages.append({"role": "assistant", "content": content})

    def build_document_block(self, content: str, title: str) -> dict[str, Any]:
        """Build a context document block. Override for provider-specific formats."""
        return {
            "type": "text",
            "text": f'<source title="{title}">\n{content}\n</source>',
        }

    def chat_json(
        self, messages: list[Any], **kwargs: Unpack[ChatParams]
    ) -> ChatResponse:
        response = self.chat(messages, **kwargs)
        response.text = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.text.strip())

        return response

    @abstractmethod
    def chat(self, messages: list[Any], **kwargs: Unpack[ChatParams]) -> ChatResponse:
        pass

    @abstractmethod
    async def chat_stream(
        self, messages: list[Any], **kwargs: Unpack[ChatParams]
    ) -> MessageStream:
        pass

    @abstractmethod
    def record_usage(self, usage: Any) -> None:
        pass

    def extract_citation_titles(self, message: Any) -> set[str]:
        return set()

    def has_web_search_results(self, raw: Any) -> bool:
        return False
