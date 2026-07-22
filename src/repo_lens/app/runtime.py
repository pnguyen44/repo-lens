from voyageai.client import Client as VoyageClient

from repo_lens.agents.agent import AgentName, create_github_agent, create_rag_agent
from repo_lens.agents.orchestrator import Orchestrator
from repo_lens.core.config import Config
from repo_lens.core.mcp_client import MCPClient
from repo_lens.core.repo_context import RepoContext
from repo_lens.providers.provider import create_chat_client
from repo_lens.providers.token_tracker import TokenTracker
from repo_lens.rag.bm25_index import BM25Index
from repo_lens.rag.chroma_index import ChromaVectorIndex
from repo_lens.rag.document_indexer import DocumentIndexer
from repo_lens.rag.embeddings import InputType, VoyageEmbedder
from repo_lens.rag.hybrid_retriever import HybridRetriever
from repo_lens.rag.indexer import fetch_repo_chunks
from repo_lens.rag.reranker import VoyageReranker

CHROMA_COLLECTION_NAME = "repo_chunks"


class App:
    def __init__(self, config: Config, github_mcp: MCPClient) -> None:
        self.config = config
        self.github_mcp = github_mcp
        self.token_tracker = TokenTracker()
        self.vector_index, self.retriever = self._create_retriever_stack()
        self.document_indexer = DocumentIndexer(
            vector_index=self.vector_index, retriever=self.retriever
        )
        self.orchestrator = self._create_orchestrator()

    def _create_retriever_stack(self) -> tuple[ChromaVectorIndex, HybridRetriever]:
        embedder = VoyageEmbedder(VoyageClient(), model=self.config.voyage_embed_model)

        vector_index = ChromaVectorIndex(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_fn=lambda texts: embedder.generate_embeddings(
                texts, input_type=InputType.DOCUMENT
            ),
            host=self.config.chroma_host,
            port=self.config.chroma_port,
        )

        bm25 = BM25Index()
        retriever = HybridRetriever(vector_index, bm25)

        return vector_index, retriever

    def _create_orchestrator(self) -> Orchestrator:
        reranker = VoyageReranker(
            client=VoyageClient(), model=self.config.voyage_rerank_model
        )

        github_chat_client = create_chat_client(
            config=self.config, token_tracker=self.token_tracker
        )
        github_agent = create_github_agent(
            chat_client=github_chat_client, github_mcp=self.github_mcp
        )

        rag_chat_client = create_chat_client(
            config=self.config, token_tracker=self.token_tracker
        )
        rag_agent = create_rag_agent(
            chat_client=rag_chat_client,
            hybrid_retriever=self.retriever,
            reranker=reranker,
        )

        orchestrator_chat_client = create_chat_client(
            config=self.config, token_tracker=self.token_tracker
        )
        return Orchestrator(
            agents={AgentName.GITHUB: github_agent, AgentName.RAG: rag_agent},
            chat_client=orchestrator_chat_client,
        )

    async def ensure_indexed(self, repo_context: RepoContext) -> None:
        if not self.document_indexer.exits(key="repo", value=repo_context.key):
            docs = await fetch_repo_chunks(
                github_mcp=self.github_mcp, repo_context=repo_context
            )
            self.document_indexer.index(docs)

    def token_summary(self) -> dict[str, object]:
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            **self.token_tracker.summary(),
        }
