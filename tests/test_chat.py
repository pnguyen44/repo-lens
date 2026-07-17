from typing import Any
from unittest.mock import MagicMock
from repo_lens.agents.chat import Chat
import pytest

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
