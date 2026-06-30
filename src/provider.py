from typing import Any
from config import Config
from chat_client import ChatClient


def create_chat_client(config: Config) -> ChatClient[Any, Any]:
    model = config.model

    if config.provider == "gemini":
        from google import genai
        from gemini_client import Gemini

        client = genai.Client()
        return Gemini(client=client, model=model)
    else:
        from anthropic import Anthropic
        from claude import Claude

        client = Anthropic()
        return Claude(client=client, model=model)
