import pytest
from bm25_index import BM25Index
from hybrid_retriever import HybridRetriever
from indexer import index_repo
from vector_index import VectorIndex
from unittest.mock import patch, AsyncMock, MagicMock
from mcp_client import MCPClient


class FakeEmbedder:
    def generate_embeddings(
        self, texts: list[str], **kwargs: object
    ) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def mock_mcp_client():
    fake_readme = (
        "# Title\n\n## Section One\nSome content.\n\n## Section Two\nMore content."
    )
    with patch(
        "indexer.fetch_readme", new_callable=AsyncMock, return_value=fake_readme
    ):
        yield None


@pytest.mark.asyncio
async def test_rag_end_to_end(mock_mcp_client, fake_embedder):
    vector_index = VectorIndex(embedding_fn=fake_embedder.generate_embeddings)
    bm25 = BM25Index()
    retriever = HybridRetriever(vector_index, bm25)
    owner = "openshift-hyperfleet"
    repo = "hyperfleet-api"

    count = await index_repo(
        mcp_client=MagicMock(spec=MCPClient),
        hybrid_retriever=retriever,
        owner=owner,
        repo=repo,
    )

    assert count == 3

    query = "how does API work?"
    results = retriever.search(query_text=query, k=2)
    assert len(results) > 0
    assert results[0][0]["repo"] == f"{owner}/{repo}"
    assert len(retriever) == 3
    assert "section" in results[0][0]


@pytest.mark.asyncio
async def test_index_repo_empty_readme(fake_embedder):
    with patch("indexer.fetch_readme", new_callable=AsyncMock, return_value=""):
        vector_index = VectorIndex(embedding_fn=fake_embedder.generate_embeddings)
        bm25 = BM25Index()
        retriever = HybridRetriever(vector_index, bm25)
        count = await index_repo(
            mcp_client=MagicMock(spec=MCPClient),
            hybrid_retriever=retriever,
            owner="org",
            repo="repo",
        )
        assert count == 0
        assert len(retriever) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "readme, expected_chunks",
    [
        ("Just a plain paragraph with no headers.", 1),
        ("# Title\n\n## \n\n## Real Section\nContent.", 2),
    ],
    ids=["no-headers", "whitespace-only-section"],
)
async def test_index_repo_chunk_counts(fake_embedder, readme, expected_chunks):
    with patch("indexer.fetch_readme", new_callable=AsyncMock, return_value=readme):
        vector_index = VectorIndex(embedding_fn=fake_embedder.generate_embeddings)
        bm25_index = BM25Index()
        retriever = HybridRetriever(vector_index, bm25_index)
        count = await index_repo(
            mcp_client=MagicMock(spec=MCPClient),
            hybrid_retriever=retriever,
            owner="org",
            repo="repo",
        )
        assert count == expected_chunks
        assert len(retriever) == expected_chunks


@pytest.mark.asyncio
async def test_index_repo_propagates_fetch_error(fake_embedder):
    with patch(
        "indexer.fetch_readme",
        new_callable=AsyncMock,
        side_effect=ValueError("No README context found"),
    ):
        vector_index = VectorIndex(embedding_fn=fake_embedder.generate_embeddings)
        bm25 = BM25Index()
        retriever = HybridRetriever(vector_index, bm25)
        with pytest.raises(ValueError, match="No README context found"):
            await index_repo(
                mcp_client=MagicMock(spec=MCPClient),
                hybrid_retriever=retriever,
                owner="org",
                repo="repo",
            )
