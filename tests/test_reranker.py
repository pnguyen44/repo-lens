from typing import Any
from unittest.mock import AsyncMock

from repo_lens.rag.reranker import VoyageReranker

MODEL = "rerank-2"
DOCS = ["doc one", "doc two"]


def _make_reranker(
    fake_results: list[Any] | None = None,
) -> tuple[AsyncMock, VoyageReranker]:
    mock_client = AsyncMock()

    if fake_results is not None:
        mock_client.rerank.return_value.results = fake_results

    return mock_client, VoyageReranker(client=mock_client, model=MODEL)


async def test_rerank_passes_correct_args() -> None:
    mock_client, reranker = _make_reranker()

    await reranker.rerank(query="test query", documents=DOCS, top_k=2)

    mock_client.rerank.assert_called_once_with(
        query="test query", documents=DOCS, model=MODEL, top_k=2
    )


async def test_rerank_returns_results() -> None:
    fake_results = [AsyncMock(index=0, document="doc one", relevance_score=0.9)]
    _, reranker = _make_reranker(fake_results=fake_results)

    result = await reranker.rerank(query="test query", documents=DOCS, top_k=2)

    assert result == fake_results


async def test_rerank_empty_docs() -> None:
    _, reranker = _make_reranker(fake_results=[])

    result = await reranker.rerank(query="test query", documents=[], top_k=2)

    assert result == []
