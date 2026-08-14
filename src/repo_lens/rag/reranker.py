import logging
from typing import Protocol, cast

from voyageai.client_async import AsyncClient as VoyageAsyncClient
from voyageai.object.reranking import RerankingResult

from repo_lens.rag.voyage import voyage_retry

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    async def rerank(
        self, query: str, documents: list[str], top_k: int = 3
    ) -> list[RerankingResult]: ...


class VoyageReranker:
    def __init__(self, client: VoyageAsyncClient, model: str) -> None:
        self.client = client
        self.model = model

    async def rerank(
        self, query: str, documents: list[str], top_k: int = 3
    ) -> list[RerankingResult]:
        reranking = await voyage_retry(
            fn=lambda: self.client.rerank(
                query=query, documents=documents, model=self.model, top_k=top_k
            ),
            retires=2,
        )
        logger.info("Rerank tokens: %d", reranking.total_tokens)
        return cast(list[RerankingResult], reranking.results)
