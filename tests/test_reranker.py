from typing import Any
from unittest.mock import MagicMock

from reranker import VoyageReranker

MODEL = "rerank-2"
DOCS = ["doc one", "doc two"]


def _make_reranker(
    fake_results: list[Any] | None = None,
) -> tuple[MagicMock, VoyageReranker]:
    mock_client = MagicMock()

    if fake_results is not None:
        mock_client.rerank.return_value.results = fake_results

    return mock_client, VoyageReranker(client=mock_client, model=MODEL)


def test_rerank_passes_correct_args() -> None:
    mock_client, reranker = _make_reranker()

    reranker.rerank(query="test query", documents=DOCS, top_k=2)

    mock_client.rerank.assert_called_once_with(
        query="test query", documents=DOCS, model=MODEL, top_k=2
    )


def test_rerank_returns_results() -> None:
    fake_results = [MagicMock(index=0, document="doc one", relevance_score=0.9)]
    _, reranker = _make_reranker(fake_results=fake_results)

    result = reranker.rerank(query="test query", documents=DOCS, top_k=2)

    assert result == fake_results


def test_rerank_empty_docs() -> None:
    _, reranker = _make_reranker(fake_results=[])

    result = reranker.rerank(query="test query", documents=[], top_k=2)

    assert result == []
