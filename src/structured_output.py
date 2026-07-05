from typing import Any, TypeVar
from chat_client import ChatClient, StreamResponse
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def parse_with_retry(
    chat_client: ChatClient[Any],
    response: StreamResponse,
    messages: list[Any],
    model_type: type[T],
    max_retries: int = 3,
) -> T:
    for attempt in range(max_retries):
        text = response.text

        try:
            return model_type.model_validate_json(text)
        except ValidationError as e:
            if attempt < max_retries - 1:
                chat_client.add_user_message(
                    messages=messages,
                    content=f"JSON validation failed: {e}\nPlease fix and return valid JSON",
                )
                response = chat_client.chat_json(messages=messages)

    raise ValueError(
        f"Failed to parse {model_type.__name__} after {max_retries} retries"
    )
