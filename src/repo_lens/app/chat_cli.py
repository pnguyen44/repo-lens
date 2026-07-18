import asyncio
import logging
import os
from contextlib import AsyncExitStack

from rich import print

from repo_lens.agents.orchestrator import delegation_label
from repo_lens.app.runtime import App
from repo_lens.core.config import Config, create_config
from repo_lens.core.logging_config import configure_logging
from repo_lens.core.mcp_client import MCPClient, create_github_client
from repo_lens.rag.indexer import fetch_repo_chunks

logger = logging.getLogger(__name__)


def print_status(label: str, message: str) -> None:
    print(f"\n[bold cyan]{label}:[/bold cyan] {message}")


async def validate_repo(github_mcp: MCPClient, owner: str, repo: str) -> bool:
    try:
        result = await github_mcp.call_tool(
            "get_file_contents", {"owner": owner, "repo": repo, "path": "."}
        )
        if result.isError:
            logger.warning("Repo %s/%s not found or not accessible.", owner, repo)
            return False
        return True
    except Exception as e:
        logger.error("Error checking repo: %s", e)
        return False


def resolve_owner(config: Config) -> str:
    if config.default_org:
        entered = input(f"> Org [{config.default_org}] (Enter to keep): ").strip()
        return entered or config.default_org
    while True:
        owner = input("> Org:").strip()
        if owner:
            return owner


async def resolve_repo(github_mcp: MCPClient, owner: str) -> str | None:
    while True:
        repo = input("> Repo name (or /back): ").strip()
        if repo.lower() == "/back":
            return None
        if await validate_repo(github_mcp=github_mcp, owner=owner, repo=repo):
            return repo


async def select_repo(github_mcp: MCPClient, config: Config) -> tuple[str, str]:
    while True:
        owner = resolve_owner(config)
        repo = await resolve_repo(github_mcp=github_mcp, owner=owner)
        if repo is None:
            continue
        return owner, repo


async def ensure_indexed(
    app: App,
    owner: str,
    repo: str,
) -> None:
    repo_key = f"{owner}/{repo}"

    if not app.document_indexer.exits(key="repo", value=repo_key):
        docs = await fetch_repo_chunks(
            github_mcp=app.github_mcp, owner=owner, repo=repo
        )
        app.document_indexer.index(docs)


async def cli_on_delegate(agent_name: str, task: str) -> None:
    print_status(label=delegation_label(agent_name), message=task)


def cli_on_tool_start(tool_name: str) -> None:
    print_status(label="Tool Call", message=tool_name)


def cli_on_tool_input(partial_json: str) -> None:
    print(partial_json, end="", flush=True)


async def chat_loop(
    app: App,
    owner: str,
    repo: str,
) -> None:
    try:
        while True:
            user_input = input("> ")
            if user_input.lower() in ("quit", "exit"):
                break
            if user_input.lower() == "/reindex":
                repo_key = f"{owner}/{repo}"
                print_status(label="Re-indexing", message=repo_key)
                docs = await fetch_repo_chunks(
                    github_mcp=app.github_mcp,
                    owner=owner,
                    repo=repo,
                )
                app.document_indexer.reindex(key="repo", value=repo_key, documents=docs)
                print_status(label="Re-indexed", message=repo_key)
                continue
            await app.orchestrator.run(
                query=user_input,
                on_delegate=cli_on_delegate,
                on_text=lambda t: print(t, end="", flush=True),
                on_tool_start=cli_on_tool_start,
                on_tool_input=cli_on_tool_input,
            )
            print()
    except KeyboardInterrupt:
        print("\nexiting")
    finally:
        print_status(label="Tokens", message=str(app.token_summary()))


async def main() -> None:
    configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    config = create_config()

    print_status(label="Provider", message=f"{config.provider}, model: {config.model}")

    async with AsyncExitStack() as stack:
        github_mcp = await stack.enter_async_context(
            create_github_client(config.github_token)
        )

        app = App(config=config, github_mcp=github_mcp)

        count = app.document_indexer.load_from_store()
        if count > 0:
            logger.info("Loaded %d chunks from persistent storage.", count)

        owner, repo = await select_repo(github_mcp=github_mcp, config=config)

        await ensure_indexed(
            app=app,
            owner=owner,
            repo=repo,
        )

        print_status(label="Chatting about", message=f"{owner}/{repo}")

        await chat_loop(
            app=app,
            owner=owner,
            repo=repo,
        )


def cli() -> None:
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.exception("Unexpected error: %s", e)


if __name__ == "__main__":
    cli()
