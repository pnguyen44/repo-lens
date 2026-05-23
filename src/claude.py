from anthropic import Anthropic

from llm import LLM


class Claude(LLM):
    def __init__(self, client: Anthropic, model: str) -> None:
        super().__init__(client, model)

    def chat(
        self,
        user_message: str,
        system: str | None = None,
        max_token: int = 1000,
        temperature: float = 1.0,
    ) -> str:
        self.add_user_message(user_message)

        params: dict[str, str | int | float] = {
            "model": self.model,
            "max_tokens": max_token,
            "temperature": temperature,
        }

        if system:
            params["system"] = system

        response = self.client.messages.create(
            **params,
            messages=self.messages,
        )

        assistant_message: str = response.content[0].text
        self.add_assistant_message(assistant_message)

        return assistant_message
