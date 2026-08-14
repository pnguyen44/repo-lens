import asyncio
import logging
from typing import Callable

from voyageai.client import Client as VoyageClient

from repo_lens.agents.agent import AgentName, create_github_agent, create_rag_agent
from repo_lens.agents.orchestrator import Orchestrator
from repo_lens.core.config import Config, VectorStore
from repo_lens.core.mcp_client import MCPClient
from repo_lens.core.repo_context import RepoContext
from repo_lens.providers.provider import create_chat_client
from repo_lens.providers.token_tracker import TokenCounts, TokenTracker
from repo_lens.rag.base_vector_index import BaseVectorIndex
from repo_lens.rag.bm25_index import BM25Index
from repo_lens.rag.chroma_index import ChromaVectorIndex
from repo_lens.rag.document_indexer import DocumentIndexer
from repo_lens.rag.embeddings import InputType, VoyageEmbedder
from repo_lens.rag.hybrid_retriever import HybridRetriever
from repo_lens.rag.indexer import EXCLUDE_FILES, RepoContentFetcher
from repo_lens.rag.qdrant_index import QdrantVectorIndex
from repo_lens.rag.reranker import VoyageReranker

logger = logging.getLogger(__name__)

COLLECTION_NAME = "repo_lens"


def _create_vector_index(
    config: Config, embedding_fn: Callable[[list[str]], list[list[float]]]
) -> BaseVectorIndex:
    if config.vector_store == VectorStore.QDRANT:
        return QdrantVectorIndex(
            collection_name=COLLECTION_NAME,
            embedding_fn=embedding_fn,
            url=config.qdrant_url or "",
            api_key=config.qdrant_api_key,
        )

    return ChromaVectorIndex(
        collection_name=COLLECTION_NAME,
        embedding_fn=embedding_fn,
        host=config.chroma_host,
        port=config.chroma_port,
    )


class App:
    def __init__(self, config: Config, github_mcp: MCPClient) -> None:
        self.config = config
        self.github_mcp = github_mcp
        self.token_tracker = TokenTracker()
        self.vector_index, self.retriever = self._create_retriever_stack()
        logger.info("Using vector store: %s", self.config.vector_store.value)
        self.document_indexer = DocumentIndexer(
            vector_index=self.vector_index,  # type: ignore[arg-type]
            retriever=self.retriever,
        )
        self.orchestrator = self._create_orchestrator()

    def _create_retriever_stack(self) -> tuple[BaseVectorIndex, HybridRetriever]:
        embedder = VoyageEmbedder(VoyageClient(), model=self.config.voyage_embed_model)

        def embedding_fn(texts: list[str]) -> list[list[float]]:
            return embedder.generate_embeddings(
                texts=texts, input_type=InputType.DOCUMENT
            )

        vector_index = _create_vector_index(
            config=self.config, embedding_fn=embedding_fn
        )

        bm25 = BM25Index()
        retriever = HybridRetriever(vector_index, bm25)  # type: ignore[arg-type]

        return vector_index, retriever

    def _create_orchestrator(self) -> Orchestrator:
        reranker = VoyageReranker(
            client=VoyageClient(), model=self.config.voyage_rerank_model
        )

        github_chat_client = create_chat_client(
            config=self.config, token_tracker=self.token_tracker
        )
        github_agent = create_github_agent(
            chat_client=github_chat_client,
            github_mcp=self.github_mcp,
            max_tool_iterations=self.config.max_tool_iterations,
        )

        rag_chat_client = create_chat_client(
            config=self.config, token_tracker=self.token_tracker
        )
        rag_agent = create_rag_agent(
            chat_client=rag_chat_client,
            hybrid_retriever=self.retriever,
            reranker=reranker,
            max_tool_iterations=self.config.max_tool_iterations,
        )

        orchestrator_chat_client = create_chat_client(
            config=self.config, token_tracker=self.token_tracker
        )
        return Orchestrator(
            agents={AgentName.GITHUB: github_agent, AgentName.RAG: rag_agent},
            chat_client=orchestrator_chat_client,
        )

    async def validate_repo(self, repo_context: RepoContext) -> bool:
        try:
            owner = repo_context.owner
            repo = repo_context.repo

            result = await self.github_mcp.call_tool(
                "get_file_contents", {"owner": owner, "repo": repo, "path": "."}
            )
            if result.isError:
                logger.warning("Repo %s/%s not found or not accessible.", owner, repo)
                return False
            return True
        except Exception as e:
            logger.error("Error checking repo: %s", e)
            return False

    async def index_file_if_needed(self, repo_context: RepoContext, path: str) -> int:
        if not path.endswith(".md"):
            return 0

        if path.split("/")[-1] in EXCLUDE_FILES:
            logger.debug("Skipping on-demand index for excluded file: %s", path)
            return 0

        if self.document_indexer.file_is_indexed(repo=repo_context.key, path=path):
            logger.debug("Skipping on-demand index, already indexed: %s", path)
            return 0

        fetcher = RepoContentFetcher(
            mcp_client=self.github_mcp, repo_context=repo_context
        )
        docs = await fetcher.fetch_file_chunks(path)
        count = await asyncio.to_thread(
            self.document_indexer.index_file,
            repo=repo_context.key,
            path=path,
            documents=docs,
        )
        logger.info("On-demand indexed %s (%d chunks)", path, count)
        return count

    async def clear_cache(self, repo_context: RepoContext) -> int:
        return await asyncio.to_thread(
            self.document_indexer.clear_repo, repo_context.key
        )

    def _format_provider_model(self) -> str:
        return f"{self.config.provider}/{self.config.model}"

    def format_token_counts(self, counts: TokenCounts) -> str:
        parts = [
            f"in {counts['input_tokens']}",
            f"out {counts['output_tokens']}",
        ]

        if "cache_read_input_tokens" in counts:
            parts.append(f"cache_read {counts['cache_read_input_tokens']}")
        if "cache_creation_input_tokens" in counts:
            parts.append(f"cache_create {counts['cache_creation_input_tokens']}")
        return ", ".join(parts)

    def format_token_summary(self, *, running_total: bool = False) -> str:
        line = (
            f"{self._format_provider_model()} | "
            f"session: {self.format_token_counts(self.token_tracker.summary())}"
        )

        if running_total:
            line += " (running total)"

        return line

    def format_tokens_for_turn(self, before: TokenCounts) -> str:
        after = self.token_tracker.summary()
        turn = TokenTracker.token_delta(before=before, after=after)
        return (
            f"{self._format_provider_model()} | "
            f"turn: {self.format_token_counts(turn)} | "
            f"session: {self.format_token_counts(after)}"
        )
