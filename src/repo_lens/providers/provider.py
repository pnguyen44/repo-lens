from typing import Any
from repo_lens.core.config import Config
from repo_lens.providers.chat_client import ChatClient


def create_chat_client(config: Config) -> ChatClient[Any]:
    model = config.model

    if config.provider == "gemini":
        from google import genai
        from repo_lens.providers.gemini import Gemini

        client = genai.Client()
        return Gemini(client=client, model=model)
    else:
        from anthropic import Anthropic
        from repo_lens.providers.claude import Claude

        client = Anthropic()
        return Claude(client=client, model=model)
