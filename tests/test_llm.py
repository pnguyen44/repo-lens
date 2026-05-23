from llm import LLM

model = "test-model"


class FakeLLM(LLM):
    def chat(
        self,
        user_message: str,
        system: str | None = None,
        max_token: int = 1000,
        temperature: float = 1.0,
    ) -> str:
        return "fake response"


def test_message_history() -> None:
    # Pass None as client since FakeLLM doesn't use it
    llm = FakeLLM(client=None, model=model)

    assert len(llm.get_chat_history()) == 0

    user_message = "What is 1 + 1?"
    llm.add_user_message(user_message)

    assert len(llm.messages) == 1
    assert llm.messages[0] == {"role": "user", "content": user_message}

    assistant_message = "1 + 1 = 2"
    llm.add_assistant_message(assistant_message)

    assert len(llm.messages) == 2
    assert llm.messages[1] == {"role": "assistant", "content": assistant_message}

    assert len(llm.get_chat_history()) == 2
