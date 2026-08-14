from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from repo_lens.core.mcp_client import MCPClient
from repo_lens.core.repo_context import RepoContext
from repo_lens.rag.bm25_index import BM25Index
from repo_lens.rag.hybrid_retriever import HybridRetriever
from repo_lens.rag.indexer import RepoContentFetcher
from repo_lens.rag.vector_index import VectorIndex


class FakeEmbedder:
    async def generate_embeddings(
        self, texts: list[str], **kwargs: object
    ) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


def make_fetcher(repo_context: RepoContext) -> RepoContentFetcher:
    return RepoContentFetcher(
        mcp_client=MagicMock(spec=MCPClient), repo_context=repo_context
    )


@pytest.fixture
def mock_single_file():
    fake_readme = (
        "# Title\n\n## Section One\nSome content.\n\n## Section Two\nMore content."
    )
    with (
        patch.object(
            RepoContentFetcher,
            "fetch_md_file_list",
            new_callable=AsyncMock,
            return_value=["README.md"],
        ),
        patch.object(
            RepoContentFetcher,
            "fetch_file",
            new_callable=AsyncMock,
            return_value=fake_readme,
        ),
    ):
        yield None


@pytest.mark.asyncio
async def test_rag_end_to_end(mock_single_file, fake_embedder):
    vector_index = VectorIndex(embedding_fn=fake_embedder.generate_embeddings)
    bm25 = BM25Index()
    retriever = HybridRetriever(vector_index, bm25)
    repo_context = RepoContext(owner="openshift-hyperfleet", repo="hyperfleet-api")

    docs = await make_fetcher(repo_context).fetch_repo_chunks()

    await retriever.add_documents(docs)

    assert len(docs) == 3

    query = "how does API work?"
    results = await retriever.search(query_text=query, k=2)
    assert len(results) > 0
    assert results[0][0]["repo"] == repo_context.key
    assert len(retriever) == 3
    assert "section" in results[0][0]


@pytest.mark.asyncio
async def test_fetch_repo_chunks_no_md_files():
    with patch.object(
        RepoContentFetcher,
        "fetch_md_file_list",
        new_callable=AsyncMock,
        return_value=[],
    ):
        docs = await make_fetcher(
            RepoContext(owner="org", repo="repo")
        ).fetch_repo_chunks()
        assert docs == []


@pytest.mark.asyncio
async def test_fetch_repo_chunks_empty_file():
    with (
        patch.object(
            RepoContentFetcher,
            "fetch_md_file_list",
            new_callable=AsyncMock,
            return_value=["README.md"],
        ),
        patch.object(
            RepoContentFetcher,
            "fetch_file",
            new_callable=AsyncMock,
            return_value="",
        ),
    ):
        docs = await make_fetcher(
            RepoContext(owner="org", repo="repo")
        ).fetch_repo_chunks()
        assert docs == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content, expected_chunks",
    [
        ("Just a plain paragraph with no headers.", 1),
        ("# Title\n\n## \n\n## Real Section\nContent.", 2),
    ],
    ids=["no-headers", "whitespace-only-section"],
)
async def test_fetch_repo_chunks_counts(content, expected_chunks):
    with (
        patch.object(
            RepoContentFetcher,
            "fetch_md_file_list",
            new_callable=AsyncMock,
            return_value=["README.md"],
        ),
        patch.object(
            RepoContentFetcher,
            "fetch_file",
            new_callable=AsyncMock,
            return_value=content,
        ),
    ):
        docs = await make_fetcher(
            RepoContext(owner="org", repo="repo")
        ).fetch_repo_chunks()
        assert len(docs) == expected_chunks


@pytest.mark.asyncio
async def test_fetch_repo_chunks_propagates_fetch_error():
    with (
        patch.object(
            RepoContentFetcher,
            "fetch_md_file_list",
            new_callable=AsyncMock,
            return_value=["README.md"],
        ),
        patch.object(
            RepoContentFetcher,
            "fetch_file",
            new_callable=AsyncMock,
            side_effect=ValueError("No context found"),
        ),
    ):
        with pytest.raises(ValueError, match="No context found"):
            await make_fetcher(
                RepoContext(owner="org", repo="repo")
            ).fetch_repo_chunks()
