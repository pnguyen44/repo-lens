import asyncio
import logging
import os
from contextlib import AsyncExitStack

from rich import print

from repo_lens.agents.orchestrator import delegation_label
from repo_lens.app.repo_selection import (
    repo_ready_message,
    resolve_repo_from_arg,
    setup_repo,
)
from repo_lens.app.runtime import App
from repo_lens.core.config import create_config
from repo_lens.core.logging_config import configure_logging
from repo_lens.core.mcp_client import create_github_client
from repo_lens.core.repo_context import RepoContext

logger = logging.getLogger(__name__)


def print_status(label: str, message: str) -> None:
    print(f"\n[bold cyan]{label}:[/bold cyan] {message}")


async def cli_on_delegate(agent_name: str, task: str) -> None:
    print_status(label=delegation_label(agent_name), message=task)


def cli_on_tool_start(tool_name: str) -> None:
    print_status(label="Tool Call", message=tool_name)


def cli_on_tool_input(partial_json: str) -> None:
    print(partial_json, end="", flush=True)


async def chat_loop(app: App, repo_context: RepoContext) -> None:
    current_repo = repo_context

    async def on_file_fetched(path: str) -> None:
        await app.index_file_if_needed(current_repo, path)

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

            match command:
                case "/clear-cache":
                    print_status(label="Clearing cache", message=current_repo.key)
                    removed = await app.clear_cache(repo_context=current_repo)
                    print_status(
                        label="Cleared",
                        message=f"{current_repo.key} ({removed} chunks removed)",
                    )
                    continue
                case "/repo":
                    arg = parts[1] if len(parts) > 1 else ""
                    new_ctx, err = await resolve_repo_from_arg(app=app, arg=arg)
                    if err is not None:
                        print(f"[red]{err}[/red]")
                        continue
                    if new_ctx is not None:
                        current_repo = new_ctx
                        print(
                            repo_ready_message(repo_key=current_repo.key, switched=True)
                        )
                    continue

            before = app.token_tracker.summary()
            await app.orchestrator.run(
                repo_context=current_repo,
                query=user_input,
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
    except KeyboardInterrupt:
        print("\nexiting")
    finally:
        print_status(
            label="Tokens",
            message=app.format_token_summary(running_total=True),
        )


async def on_status(label: str, message: str) -> None:
    print_status(label=label, message=message)


async def main() -> None:
    configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    config = create_config()

    print_status(label="Provider", message=f"{config.provider}, model: {config.model}")

    async with AsyncExitStack() as stack:
        github_mcp = await stack.enter_async_context(
            create_github_client(config.github_token)
        )

        app = App(config=config, github_mcp=github_mcp)

        repo_context = await setup_repo(app=app, config=config, on_status=on_status)
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
