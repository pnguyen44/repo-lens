import asyncio
import logging
import os
from contextlib import AsyncExitStack
from typing import Any

from rich import print
from voyageai.client import Client as VoyageClient

from agent import AgentName, create_github_agent, create_rag_agent
from bm25_index import BM25Index
from chat_client import ChatClient
from chroma_index import ChromaVectorIndex
from config import Config, create_config
from document_indexer import DocumentIndexer
from embeddings import InputType, VoyageEmbedder
from hybrid_retriever import HybridRetriever
from indexer import fetch_repo_chunks
from logging_config import configure_logging
from mcp_client import MCPClient, create_github_client
from orchestrator import Orchestrator
from provider import create_chat_client
from reranker import VoyageReranker

CHROMA_COLLECTION_NAME = "repo_chunks"

logger = logging.getLogger(__name__)


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


async def cli_on_delegate(agent_name: str, task: str) -> None:
    print(f"\n[Delegating to {agent_name}]: {task}")


async def chat_loop(
    github_mcp: MCPClient,
    owner: str,
    repo: str,
    document_indexer: DocumentIndexer,
    orchestrator: Orchestrator,
    chat_client: ChatClient[Any],
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
            await orchestrator.run(query=user_input, on_delegate=cli_on_delegate)
    except KeyboardInterrupt:
        print("\nexiting")
    finally:
        print(chat_client.token_tracker.summary())


def create_retriever_stack(config: Config) -> tuple[ChromaVectorIndex, HybridRetriever]:
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

    return vector_index, retriever


def create_orchestrator(
    config: Config,
    chat_client: ChatClient[Any],
    retriever: HybridRetriever,
    github_mcp: MCPClient,
) -> Orchestrator:
    reranker = VoyageReranker(client=VoyageClient(), model=config.voyage_rerank_model)

    github_agent = create_github_agent(chat_client=chat_client, github_mcp=github_mcp)
    rag_agent = create_rag_agent(
        chat_client=chat_client, hybrid_retriever=retriever, reranker=reranker
    )

    return Orchestrator(
        agents={AgentName.GITHUB: github_agent, AgentName.RAG: rag_agent},
        chat_client=chat_client,
    )


async def main() -> None:
    configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    config = create_config()

    print(f"Chatting with provider: {config.provider}, model: {config.model}")

    async with AsyncExitStack() as stack:
        github_mcp = await stack.enter_async_context(
            create_github_client(config.github_token)
        )

        owner = config.default_org or input("> GITHUB org: ")
        vector_index, retriever = create_retriever_stack(config)
        document_indexer = DocumentIndexer(
            vector_index=vector_index, retriever=retriever
        )

        count = document_indexer.load_from_store()
        if count > 0:
            logger.info("Loaded %d chunks from persistent storage.", count)

        repo = await ensure_indexed(
            github_mcp=github_mcp, owner=owner, document_indexer=document_indexer
        )

        print(f"Chatting about {owner}/{repo}")

        chat_client = create_chat_client(config=config)

        orchestrator = create_orchestrator(
            config=config,
            chat_client=chat_client,
            retriever=retriever,
            github_mcp=github_mcp,
        )

        await chat_loop(
            github_mcp=github_mcp,
            owner=owner,
            repo=repo,
            document_indexer=document_indexer,
            orchestrator=orchestrator,
            chat_client=chat_client,
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
