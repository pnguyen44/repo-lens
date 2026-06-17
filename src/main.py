import logging
import os
from contextlib import AsyncExitStack
from config import create_config
from claude import Claude
from anthropic import Anthropic
from rich import print
from indexer import index_repo
from mcp_client import MCPClient, create_github_client
import asyncio
from chat import Chat
from voyageai.client import Client as VoyageClient
from embeddings import Embedder, VoyageEmbedder
from reranker import VoyageReranker
from vector_index import VectorIndex

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are repo-lens, a GitHub repository assistant.
{org_context}
When the user asks about PRs, issues, or repos across the org, use the search tool
(e.g., "is:pr is:open org:<org>") instead of listing repos individually.
When listing issues or PRs, include number, title, status, and assignee.
When answering using context from <source> tags, cite the repo and section in your response.
Be concise.

If the context says no relevant information was found, you MUST immediately search the repository
using available tools before responding. Never ask the user whether to search. Only say you don't
have enough information after tools also return nothing useful.

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


async def prompt_and_index(
    github_mcp: MCPClient, embedder: Embedder, index: VectorIndex, owner: str
) -> str:
    while True:
        repo = input("> Repo name: ")
        if not await validate_repo(github_mcp=github_mcp, owner=owner, repo=repo):
            continue
        try:
            await index_repo(
                mcp_client=github_mcp,
                embedder=embedder,
                index=index,
                owner=owner,
                repo=repo,
            )
        except Exception as e:
            logger.error("Could not index README: %s/%s: %s", owner, repo, e)
            logger.warning("Chat will work without RAG context.")

        return repo


async def main() -> None:
    config = create_config()
    client = Anthropic()

    async with AsyncExitStack() as stack:
        github_mcp = await stack.enter_async_context(
            create_github_client(config.github_token)
        )

        owner = config.default_org or input("> GITHUB org: ")

        claude = Claude(client=client, model=config.claude_model)

        embedder = VoyageEmbedder(VoyageClient(), model=config.voyage_embed_model)
        index = VectorIndex()

        repo = await prompt_and_index(
            github_mcp=github_mcp, embedder=embedder, index=index, owner=owner
        )

        print(f"Chatting about {owner}/{repo}")

        reranker = VoyageReranker(
            client=VoyageClient(), model=config.voyage_rerank_model
        )

        chat = Chat(
            chat_client=claude,
            mcp_clients={"github": github_mcp},
            system_prompt=build_system_prompt(config.default_org),
            embedder=embedder,
            index=index,
            reranker=reranker,
        )

        try:
            while True:
                user_input = input("> ")
                if user_input.lower() in ("quit", "exit"):
                    break
                await chat.run(user_input)
        except KeyboardInterrupt:
            print("\nexiting")
        finally:
            print(claude.token_tracker.summary())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error("Unexpected error: %s", e)
