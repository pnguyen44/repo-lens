import logging
import os
from contextlib import AsyncExitStack
from config import create_config
from rich import print
from indexer import index_repo
from mcp_client import MCPClient, create_github_client
import asyncio
from chat import Chat
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


async def prompt_and_index(
    github_mcp: MCPClient,
    retriever: HybridRetriever,
    owner: str,
    vector_index: ChromaVectorIndex,
) -> str:
    while True:
        repo = input("> Repo name: ")
        if not await validate_repo(github_mcp=github_mcp, owner=owner, repo=repo):
            continue
        try:
            repo_key = f"{owner}/{repo}"
            if vector_index.exists_in_collection("repo", repo_key):
                print(f"'{repo_key}' already indexed, skipping.")
            else:
                await index_repo(
                    mcp_client=github_mcp,
                    hybrid_retriever=retriever,
                    owner=owner,
                    repo=repo,
                )
        except Exception as e:
            logger.error("Could not index README: %s/%s: %s", owner, repo, e)
            logger.warning("Chat will work without RAG context.")

        return repo


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
        stored_docs = vector_index.get_all_documents()

        if stored_docs:
            bm25.add_documents(stored_docs)
            logger.info("Loaded %d chunks from persistent storage.", len(stored_docs))

        retriever = HybridRetriever(vector_index, bm25)

        repo = await prompt_and_index(
            github_mcp=github_mcp,
            retriever=retriever,
            owner=owner,
            vector_index=vector_index,
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

        try:
            while True:
                user_input = input("> ")
                if user_input.lower() in ("quit", "exit"):
                    break
                await chat.run(user_input)
        except KeyboardInterrupt:
            print("\nexiting")
        finally:
            print(client.token_tracker.summary())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error("Unexpected error: %s", e)
