from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from repo_lens.app.runtime import App
from repo_lens.core.config import Config, VectorStore
from repo_lens.core.repo_context import RepoContext
from repo_lens.providers.token_tracker import TokenTracker
from repo_lens.rag.types import FetchedFile


def _make_app(*, provider: str = "gemini", model: str = "gemini-2.5-flash") -> App:
    app = App.__new__(App)
    app.config = Config(
        provider=provider,
        model=model,
        github_token="test-token",
        default_org=None,
        default_repo=None,
        voyage_embed_model="voyage-3-large",
        voyage_rerank_model="rerank-2",
        chroma_host="localhost",
        chroma_port=8000,
        vector_store=VectorStore.QDRANT,
        qdrant_url=None,
        qdrant_api_key=None,
    )
    app.token_tracker = TokenTracker()
    app.github_mcp = MagicMock()
    return app


def test_format_token_counts_basic() -> None:
    app = _make_app()
    assert (
        app.format_token_counts(
            {"input_tokens": 100, "output_tokens": 50, "request_count": 1}
        )
        == "in 100, out 50"
    )


def test_format_token_counts_with_cache() -> None:
    app = _make_app()
    assert (
        app.format_token_counts(
            {
                "input_tokens": 100,
                "output_tokens": 50,
                "request_count": 1,
                "cache_read_input_tokens": 20,
                "cache_creation_input_tokens": 10,
            }
        )
        == "in 100, out 50, cache_read 20, cache_create 10"
    )


def test_format_tokens_for_turn() -> None:
    app = _make_app()
    before = app.token_tracker.summary()
    app.token_tracker.record({"input_tokens": 100, "output_tokens": 50})

    assert (
        app.format_tokens_for_turn(before)
        == "gemini/gemini-2.5-flash | turn: in 100, out 50 | session: in 100, out 50"
    )


def test_format_token_summary_running_total() -> None:
    app = _make_app(provider="anthropic", model="claude-sonnet")
    app.token_tracker.record({"input_tokens": 10, "output_tokens": 5})

    assert (
        app.format_token_summary(running_total=True)
        == "anthropic/claude-sonnet | session: in 10, out 5 (running total)"
    )


# --- index_file_if_needed SHA tests ---

REPO_CONTEXT = RepoContext(owner="org", repo="repo")


@pytest.fixture
def app_with_indexer() -> Any:
    app = _make_app()
    app.document_indexer = AsyncMock()
    return app


@pytest.mark.parametrize(
    "indexed_sha, fetched_sha, expected_count, expect_evict, expect_index",
    [
        pytest.param("sha_aaa", "sha_aaa", 0, False, False, id="sha-match-skips"),
        pytest.param("sha_old", "sha_new", 3, True, True, id="sha-mismatch-evicts"),
        pytest.param(None, "sha_first", 2, False, True, id="first-time-indexes"),
    ],
)
async def test_index_file_if_needed_sha_paths(
    app_with_indexer: Any,
    indexed_sha: str | None,
    fetched_sha: str,
    expected_count: int,
    expect_evict: bool,
    expect_index: bool,
) -> None:
    app = app_with_indexer
    app.document_indexer.get_indexed_sha.return_value = indexed_sha
    app.document_indexer.index_file.return_value = expected_count

    with patch("repo_lens.app.runtime.RepoContentFetcher") as MockFetcher:
        instance = MockFetcher.return_value
        instance.fetch_file = AsyncMock(
            return_value=FetchedFile(text="# Content", sha=fetched_sha)
        )

        count = await app.index_file_if_needed(REPO_CONTEXT, "docs/readme.md")

    assert count == expected_count

    if expect_evict:
        app.document_indexer.evict_file.assert_awaited_once_with(
            repo="org/repo", path="docs/readme.md"
        )
    else:
        app.document_indexer.evict_file.assert_not_awaited()

    if expect_index:
        app.document_indexer.index_file.assert_awaited_once()
    else:
        app.document_indexer.index_file.assert_not_awaited()


async def test_index_file_if_needed_non_md_skips(app_with_indexer: Any) -> None:
    """Non-markdown files are always skipped."""
    app = app_with_indexer
    count = await app.index_file_if_needed(REPO_CONTEXT, "src/main.py")

    assert count == 0
    app.document_indexer.get_indexed_sha.assert_not_awaited()
