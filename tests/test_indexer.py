import pytest
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
    embedder = fake_embedder
    index = VectorIndex()
    owner = "openshift-hyperfleet"
    repo = "hyperfleet-api"

    count = await index_repo(
        mcp_client=MagicMock(spec=MCPClient),
        embedder=embedder,
        index=index,
        owner=owner,
        repo=repo,
    )

    assert count == 3

    query = "how does API work?"
    query_vector = embedder.generate_embeddings([query])[0]
    k = 2
    results = index.search(query=query_vector, k=k)
    assert len(results) > 0
    assert results[0][0]["repo"] == f"{owner}/{repo}"
    assert len(index) == 3
    assert "section" in results[0][0]


@pytest.mark.asyncio
async def test_index_repo_empty_readme(fake_embedder):
    with patch("indexer.fetch_readme", new_callable=AsyncMock, return_value=""):
        index = VectorIndex()
        count = await index_repo(
            mcp_client=MagicMock(spec=MCPClient),
            embedder=fake_embedder,
            index=index,
            owner="org",
            repo="repo",
        )
        assert count == 0
        assert len(index) == 0


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
        index = VectorIndex()
        count = await index_repo(
            mcp_client=MagicMock(spec=MCPClient),
            embedder=fake_embedder,
            index=index,
            owner="org",
            repo="repo",
        )
        assert count == expected_chunks
        assert len(index) == expected_chunks


@pytest.mark.asyncio
async def test_index_repo_propagates_fetch_error(fake_embedder):
    with patch(
        "indexer.fetch_readme",
        new_callable=AsyncMock,
        side_effect=ValueError("No README context found"),
    ):
        index = VectorIndex()
        with pytest.raises(ValueError, match="No README context found"):
            await index_repo(
                mcp_client=MagicMock(spec=MCPClient),
                embedder=fake_embedder,
                index=index,
                owner="org",
                repo="repo",
            )
