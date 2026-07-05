from typing import Any

import pytest
from dotenv import load_dotenv
from google import genai
from gemini import Gemini

load_dotenv()

model = "gemini-2.5-flash"


@pytest.fixture  # type: ignore[misc]
def gemini() -> Gemini:
    client = genai.Client()
    return Gemini(client=client, model=model)


def test_basic_response(gemini: Gemini) -> None:
    messages: list[Any] = []
    gemini.add_user_message(messages, "Reply with exactly the word 'hello'")
    response = gemini.chat(messages, system="You are a helpful assistant.")

    assert response.text is not None
    assert len(response.text) > 0


def test_multi_turn(gemini: Gemini) -> None:
    messages: list[Any] = []
    gemini.add_user_message(messages, "Remember the number 42")
    response = gemini.chat(messages, system="You are a helpful assistant.")

    gemini.add_assistant_message(messages, response)
    gemini.add_user_message(messages, "What number did I just tell you?")
    response2 = gemini.chat(messages, system="You are a helpful assistant.")

    assert "42" in response2.text
