import logging
import os
from contextlib import AsyncExitStack
from typing import Any
from config import create_config
from rich import print
from document_indexer import DocumentIndexer
from indexer import fetch_repo_chunks
from mcp_client import MCPClient, create_github_client
import asyncio
from chat import Chat
from chat_client import ChatClient
from voyageai.client import Client as VoyageClient
from embeddings import VoyageEmbedder, InputType
from reranker import VoyageReranker
from bm25_index import BM25Index
from hybrid_retriever import HybridRetriever
from provider import create_chat_client
from chroma_index import ChromaVectorIndex


CHROMA_COLLECTION_NAME = "repo_chunks"

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


async def resolve_repo(github_mcp: MCPClient, owner: str) -> str:
    while True:
        repo = input("> Repo name: ")
        if await validate_repo(github_mcp=github_mcp, owner=owner, repo=repo):
            return repo


async def ensure_indexed(
    github_mcp: MCPClient, owner: str, document_indexer: DocumentIndexer
) -> str:
    repo = await resolve_repo(github_mcp=github_mcp, owner=owner)

    repo_key = f"{owner}/{repo}"

    if not document_indexer.exits(key="repo", value=repo_key):
        docs = await fetch_repo_chunks(github_mcp=github_mcp, owner=owner, repo=repo)

        document_indexer.index(docs)
    return repo


async def chat_loop(
    chat: Chat,
    github_mcp: MCPClient,
    owner: str,
    repo: str,
    document_indexer: DocumentIndexer,
    client: ChatClient[Any],
) -> None:
    try:
        while True:
            user_input = input("> ")
            if user_input.lower() in ("quit", "exit"):
                break
            if user_input.lower() == "/reindex":
                repo_key = f"{owner}/{repo}"
                print(f"Re-indexing {repo_key}...")
                docs = await fetch_repo_chunks(
                    github_mcp=github_mcp,
                    owner=owner,
                    repo=repo,
                )
                document_indexer.reindex(key="repo", value=repo_key, documents=docs)
                print(f"Re-indexed {repo_key} successfully.")
                continue
            await chat.run(user_input)
    except KeyboardInterrupt:
        print("\nexiting")
    finally:
        print(client.token_tracker.summary())


async def main() -> None:
    config = create_config()

    async with AsyncExitStack() as stack:
        github_mcp = await stack.enter_async_context(
            create_github_client(config.github_token)
        )

        owner = config.default_org or input("> GITHUB org: ")

        client = create_chat_client(config=config)

        embedder = VoyageEmbedder(VoyageClient(), model=config.voyage_embed_model)

        vector_index = ChromaVectorIndex(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_fn=lambda texts: embedder.generate_embeddings(
                texts, input_type=InputType.DOCUMENT
            ),
            host=config.chroma_host,
            port=config.chroma_port,
        )
        bm25 = BM25Index()

        retriever = HybridRetriever(vector_index, bm25)

        document_indexer = DocumentIndexer(
            vector_index=vector_index, retriever=retriever
        )

        count = document_indexer.load_from_store()

        if count > 0:
            logger.info("Loaded %d chunks from persistent storage.", count)

        repo = await ensure_indexed(
            github_mcp=github_mcp, owner=owner, document_indexer=document_indexer
        )

        print(
            f"Chatting about {owner}/{repo} (provider: {config.provider}, model: {config.model})"
        )

        reranker = VoyageReranker(
            client=VoyageClient(), model=config.voyage_rerank_model
        )

        chat = Chat(
            chat_client=client,
            mcp_clients={"github": github_mcp},
            system_prompt=build_system_prompt(config.default_org),
            embedder=embedder,
            hybrid_retriever=retriever,
            reranker=reranker,
        )

        await chat_loop(
            chat=chat,
            github_mcp=github_mcp,
            owner=owner,
            repo=repo,
            document_indexer=document_indexer,
            client=client,
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error("Unexpected error: %s", e)
