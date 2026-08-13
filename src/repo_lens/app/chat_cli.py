import asyncio
import logging
import os
from contextlib import AsyncExitStack

from anthropic import RateLimitError as AnthropicRateLimitError
from google.genai.errors import ClientError as GeminiClientError
from rich import print

from repo_lens.agents.orchestrator import delegation_label
from repo_lens.app.repo_selection import (
    ChatRepoState,
    handle_command,
    setup_repo,
)
from repo_lens.app.runtime import App
from repo_lens.core.config import create_config
from repo_lens.core.logging_config import configure_logging
from repo_lens.core.mcp_client import create_github_client
from repo_lens.core.repo_context import RepoContext
from repo_lens.core.retry import format_rate_limit_message

logger = logging.getLogger(__name__)


async def print_message(message: str, *, error: bool = False) -> None:
    if error:
        print(f"[red]{message}[/red]")
    else:
        print(message)


def print_status(label: str, message: str) -> None:
    print(f"\n[bold cyan]{label}:[/bold cyan] {message}")


async def cli_on_delegate(agent_name: str, task: str) -> None:
    print_status(label=delegation_label(agent_name), message=task)


def cli_on_tool_start(tool_name: str) -> None:
    print_status(label="Tool Call", message=tool_name)


def cli_on_tool_input(partial_json: str) -> None:
    print(partial_json, end="", flush=True)


async def run_chat_turn(app: App, state: ChatRepoState, query: str) -> None:
    async def on_file_fetched(path: str) -> None:
        await app.index_file_if_needed(state.repo_context, path)

    before = app.token_tracker.summary()
    await app.orchestrator.run(
        repo_context=state.repo_context,
        query=query,
        on_delegate=cli_on_delegate,
        on_text=lambda t: print(t, end="", flush=True),
        on_tool_start=cli_on_tool_start,
        on_tool_input=cli_on_tool_input,
        on_file_fetched=on_file_fetched,
    )
    print()
    print_status(
        label="Tokens",
        message=app.format_tokens_for_turn(before),
    )


async def chat_loop(app: App, repo_context: RepoContext) -> None:
    state = ChatRepoState.from_context(repo_context)

    try:
        while True:
            user_input = input("> ")
            stripped = user_input.strip()

            if not stripped:
                continue

            if stripped.lower() in ("quit", "exit"):
                break

            parts = stripped.split(maxsplit=1)
            command = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if await handle_command(
                app=app,
                state=state,
                command=command,
                arg=arg,
                on_message=print_message,
            ):
                continue

            try:
                await run_chat_turn(app=app, state=state, query=user_input)
            except GeminiClientError as e:
                if e.code != 429:
                    raise
                detail = getattr(e, "message", str(e))
                await print_message(format_rate_limit_message(detail), error=True)
            except AnthropicRateLimitError:
                await print_message(format_rate_limit_message("rate limit"), error=True)

    except KeyboardInterrupt:
        print("\nexiting")
    finally:
        print_status(
            label="Tokens",
            message=app.format_token_summary(running_total=True),
        )


async def main() -> None:
    configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    config = create_config()

    print_status(label="Provider", message=f"{config.provider}, model: {config.model}")

    async with AsyncExitStack() as stack:
        github_mcp = await stack.enter_async_context(
            create_github_client(config.github_token)
        )

        app = App(config=config, github_mcp=github_mcp)

        repo_context = await setup_repo(
            app=app, config=config, on_message=print_message
        )
        if repo_context is None:
            return

        await chat_loop(app=app, repo_context=repo_context)


def cli() -> None:
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.exception("Unexpected error: %s", e)


if __name__ == "__main__":
    cli()
