from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from anthropic import RateLimitError as AnthropicRateLimitError
from google.genai.errors import ClientError as GeminiClientError

from repo_lens.agents.chat import Chat
from repo_lens.agents.tool_manager import ToolManager
from repo_lens.core.repo_context import RepoContext
from repo_lens.providers.chat_client import ChatResponse, StreamChunk, ToolCall
from repo_lens.rag.hybrid_retriever import HybridRetriever
from repo_lens.rag.vector_index import VectorIndex


def _fake_embed(texts: list[str]) -> list[list[float]]:
    return [[1.0, 0.0, 0.0] for _ in texts]


def _make_chat(index: VectorIndex) -> Chat:
    mock_client = MagicMock()
    mock_client.build_document_block.side_effect = lambda content, title: {
        "type": "text",
        "text": f'<source title="{title}">\n{content}\n</source>',
    }

    retriever = HybridRetriever(index)

    return Chat(chat_client=mock_client, mcp_clients={}, hybrid_retriever=retriever)


def _flatten_content(result: str | list[Any]) -> str:
    if isinstance(result, str):
        return result

    return "\n".join(block["text"] for block in result if "text" in block)


CONTEXT_CASES = [
    {
        "name": "includes relevant chunks",
        "vectors": [
            (
                [1.0, 0.0, 0.0],
                {
                    "content": "good",
                    "repo": "r",
                    "section": "s",
                    "url": "http://example.com",
                },
            )
        ],
        "expected_in": ["good", "<source"],
        "expected_not_in": [],
    },
    {
        "name": "wraps single chunk in document block",
        "vectors": [
            (
                [0.0, 1.0, 0.0],
                {
                    "content": "bad",
                    "repo": "r",
                    "section": "s",
                    "url": "http://example.com",
                },
            )
        ],
        "expected_in": ["bad", "<source"],
        "expected_not_in": [],
    },
    {
        "name": "wraps multiple chunks in document blocks",
        "vectors": [
            (
                [1.0, 0.0, 0.0],
                {
                    "content": "good",
                    "repo": "r",
                    "section": "s",
                    "url": "http://example.com",
                },
            ),
            (
                [0.0, 1.0, 0.0],
                {
                    "content": "bad",
                    "repo": "r",
                    "section": "s",
                    "url": "http://example.com",
                },
            ),
        ],
        "expected_in": ["good", "bad", "<source"],
        "expected_not_in": [],
    },
]


@pytest.mark.parametrize("case", CONTEXT_CASES, ids=lambda c: c["name"])
def test_build_context(case) -> None:
    index = VectorIndex(embedding_fn=_fake_embed)
    for vec, doc in case["vectors"]:
        index.add_vector(vec, doc)

    chat = _make_chat(index)
    result = chat._build_context("test query")

    print("result", result)

    flat = _flatten_content(result)

    for item in case["expected_in"]:
        assert item in flat

    for item in case["expected_not_in"]:
        assert item not in flat


def test_build_context_empty_index() -> None:
    index = VectorIndex(embedding_fn=_fake_embed)
    chat = _make_chat(index)
    result = chat._build_context("test query")

    assert result == ""


def test_build_context_filters_by_repo_context_key() -> None:
    retriever = MagicMock()
    retriever.search.return_value = []

    chat = Chat(chat_client=MagicMock(), hybrid_retriever=retriever)
    chat.repo_context = RepoContext(owner="org", repo="my-repo")

    chat._build_context("test query")

    retriever.search.assert_called_once_with(
        query_text="test query", k=3, repo="org/my-repo"
    )


def test_build_context_without_repo_context_passes_none() -> None:
    retriever = MagicMock()
    retriever.search.return_value = []

    chat = Chat(chat_client=MagicMock(), hybrid_retriever=retriever)

    chat._build_context("test query")

    retriever.search.assert_called_once_with(query_text="test query", k=3, repo=None)


def _make_stream(
    response: ChatResponse, chunks: list[StreamChunk] | None = None
) -> MagicMock:
    chunk_list = chunks or []
    stream = MagicMock()
    stream.__enter__.return_value = stream
    stream.__exit__.return_value = None
    stream.__iter__.side_effect = lambda: iter(chunk_list)
    stream.get_final_message.return_value = response
    return stream


@pytest.mark.asyncio
async def test_run_stops_at_max_tool_iterations() -> None:
    tool_call_response = ChatResponse(
        stop_reason="tool_use",
        text="",
        tool_calls=[ToolCall(id="1", name="foo", input={})],
        raw={"content": []},
    )

    chat_client = MagicMock()
    chat_client.chat_stream.return_value = _make_stream(tool_call_response)

    chat = Chat(
        chat_client=chat_client,
        mcp_clients={},
        max_tool_iterations=2,
    )
    chat.tools = [{"name": "foo"}]

    with (
        patch.object(ToolManager, "get_all_tools", new=AsyncMock(return_value=[])),
        patch.object(
            ToolManager,
            "execute_tool_requests",
            new=AsyncMock(return_value=[]),
        ) as execute_tools,
    ):
        await chat.run("test query")

    assert chat_client.chat_stream.call_count == 3
    assert execute_tools.await_count == 2


@pytest.mark.asyncio
async def test_run_raises_gemini_429_after_retries_exhausted() -> None:
    chat_client = MagicMock()
    chat_client.chat_stream.side_effect = GeminiClientError(
        429, {"error": {"message": "Quota exceeded. Please retry in 0s."}}
    )

    chat = Chat(chat_client=chat_client, mcp_clients={})

    with patch.object(ToolManager, "get_all_tools", new=AsyncMock(return_value=[])):
        with pytest.raises(GeminiClientError) as exc_info:
            await chat.run("test query")

    assert exc_info.value.code == 429


@pytest.mark.asyncio
async def test_run_raises_anthropic_rate_limit_after_retries_exhausted() -> None:
    response = httpx.Response(
        429, request=httpx.Request("POST", "https://api.anthropic.com")
    )

    chat_client = MagicMock()
    chat_client.chat_stream.side_effect = AnthropicRateLimitError(
        "rate limited", response=response, body=None
    )

    chat = Chat(chat_client=chat_client, mcp_clients={})

    with patch.object(ToolManager, "get_all_tools", new=AsyncMock(return_value=[])):
        with pytest.raises(AnthropicRateLimitError):
            await chat.run("test query")
