import pytest
from bm25_index import BM25Index
from hybrid_retriever import HybridRetriever
from indexer import fetch_repo_chunks
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

    docs = await fetch_repo_chunks(
        github_mcp=MagicMock(spec=MCPClient),
        owner=owner,
        repo=repo,
    )

    retriever.add_documents(docs)

    assert len(docs) == 3

    query = "how does API work?"
    results = retriever.search(query_text=query, k=2)
    assert len(results) > 0
    assert results[0][0]["repo"] == f"{owner}/{repo}"
    assert len(retriever) == 3
    assert "section" in results[0][0]


@pytest.mark.asyncio
async def test_fetch_repo_chunks_empty_readme():
    with patch("indexer.fetch_readme", new_callable=AsyncMock, return_value=""):
        docs = await fetch_repo_chunks(
            github_mcp=MagicMock(spec=MCPClient),
            owner="org",
            repo="repo",
        )
        assert docs == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "readme, expected_chunks",
    [
        ("Just a plain paragraph with no headers.", 1),
        ("# Title\n\n## \n\n## Real Section\nContent.", 2),
    ],
    ids=["no-headers", "whitespace-only-section"],
)
async def test_fetch_repo_chunks_counts(readme, expected_chunks):
    with patch("indexer.fetch_readme", new_callable=AsyncMock, return_value=readme):
        docs = await fetch_repo_chunks(
            github_mcp=MagicMock(spec=MCPClient),
            owner="org",
            repo="repo",
        )
        assert len(docs) == expected_chunks


@pytest.mark.asyncio
async def test_fetch_repo_chunks_propagates_fetch_error():
    with patch(
        "indexer.fetch_readme",
        new_callable=AsyncMock,
        side_effect=ValueError("No README context found"),
    ):
        with pytest.raises(ValueError, match="No README context found"):
            await fetch_repo_chunks(
                github_mcp=MagicMock(spec=MCPClient),
                owner="org",
                repo="repo",
            )
