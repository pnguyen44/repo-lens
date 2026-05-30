from contextlib import AsyncExitStack
from config import create_config
from claude import Claude
from anthropic import Anthropic
from rich import print
from mcp_client import create_github_client
import asyncio
from chat import Chat

SYSTEM_PROMPT = """You are repo-lens, a GitHub repository assistant.
{org_context}
When the user asks about PRs, issues, or repos across the org, use the search tool
(e.g., "is:pr is:open org:<org>") instead of listing repos individually.
When listing issues or PRs, include number, title, status, and assignee.
Be concise.
"""


def build_system_prompt(default_org: str | None) -> str:
    if default_org:
        org_context = (
            f"The default GitHub organization is {default_org}. "
            f"When a user mentions a repo without specifying the owner, "
            f"assume the owner is {default_org} (e.g., {default_org}/repo-name). "
            f"If the user does not specify a repo name, ask them to clarify."
        )
    else:
        org_context = (
            "When a user asks about a repo without specifying the owner, "
            "ask them to clarify the organization."
        )
    return SYSTEM_PROMPT.format(org_context=org_context)


async def main() -> None:
    config = create_config()
    client = Anthropic()

    async with AsyncExitStack() as stack:
        github_client = await stack.enter_async_context(
            create_github_client(config.github_token)
        )

        claude = Claude(client=client, model=config.claude_model)
        chat = Chat(
            chat_client=claude,
            mcp_clients={"github": github_client},
            system_prompt=build_system_prompt(config.default_org),
        )

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
