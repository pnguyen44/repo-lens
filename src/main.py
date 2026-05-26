from contextlib import AsyncExitStack
from config import create_config
from claude import Claude
from anthropic import Anthropic
from rich import print
from create_github_client import create_github_client
import asyncio
from chat import Chat


async def main() -> None:
    config = create_config()
    client = Anthropic()

    async with AsyncExitStack() as stack:
        github_client = await stack.enter_async_context(
            create_github_client(config.github_token)
        )

        claude = Claude(client=client, model=config.claude_model)
        chat = Chat(chat_client=claude, mcp_clients={"github": github_client})

        while True:
            user_input = input("> ")
            if user_input.lower() in ("quit", "exit"):
                break
            await chat.run(user_input)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n exiting")
