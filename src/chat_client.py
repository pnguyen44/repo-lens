from abc import ABC, abstractmethod
from typing import Any, Generic, Iterator, Protocol, Self, TypedDict, TypeVar, Unpack

from token_tracker import TokenTracker


class MessageStream(Protocol):
    def __enter__(self) -> Self: ...
    def __exit__(self, *args: Any) -> None: ...
    def __iter__(self) -> Iterator[Any]: ...
    def __next__(self) -> Any: ...
    def get_final_message(self) -> Any: ...


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
# R = the response type (Message, ChatCompletion, etc.)
T = TypeVar("T")
R = TypeVar("R")


class ChatClient(ABC, Generic[T, R]):
    def __init__(self, client: T, model: str) -> None:
        self.client = client
        self.model = model
        self.token_tracker = TokenTracker()

    def add_user_message(self, messages: list[Any], content: str | list[Any]) -> None:
        messages.append({"role": "user", "content": content})

    def add_assistant_message(self, messages: list[Any], message: R | str) -> None:
        if isinstance(message, str):
            content: Any = message
        else:
            content = message.content  # type: ignore[attr-defined]
        messages.append({"role": "assistant", "content": content})

    def build_document_block(self, content: str, title: str) -> dict[str, Any]:
        """Build a context document block. Override for provider-specific formats."""
        return {
            "type": "text",
            "text": f'<source title="{title}">\n{content}\n</source>',
        }

    @abstractmethod
    def chat(self, messages: list[Any], **kwargs: Unpack[ChatParams]) -> R:
        pass

    @abstractmethod
    def chat_stream(
        self, messages: list[Any], **kwargs: Unpack[ChatParams]
    ) -> MessageStream:
        pass

    @abstractmethod
    def text_from_message(self, message: Any) -> str:
        pass

    @abstractmethod
    def record_usage(self, usage: Any) -> None:
        pass

    def extract_citation_titles(self, message: Any) -> set[str]:
        return set()
