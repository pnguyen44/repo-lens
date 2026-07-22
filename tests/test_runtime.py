from unittest.mock import MagicMock

from repo_lens.app.runtime import App
from repo_lens.core.config import Config
from repo_lens.providers.token_tracker import TokenTracker


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
