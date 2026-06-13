from typing import Any
from unittest.mock import MagicMock
from chat import Chat
import pytest

from vector_index import VectorIndex


def _make_chat(index: VectorIndex) -> Chat:
    embedder = MagicMock()
    embedder.generate_embeddings.return_value = [[1.0, 0.0, 0.0]]

    mock_client = MagicMock()
    mock_client.build_document_block.side_effect = lambda content, title: {
        "type": "text",
        "text": f'<source title="{title}">\n{content}\n</source>',
    }

    return Chat(chat_client=mock_client, mcp_clients={}, embedder=embedder, index=index)


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
        "name": "excludes irrelevant chunks",
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
        "expected_in": [],
        "expected_not_in": ["bad"],
    },
    {
        "name": "filters mixed relevance",
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
        "expected_in": ["good", "<source"],
        "expected_not_in": ["bad"],
    },
]


@pytest.mark.parametrize("case", CONTEXT_CASES, ids=lambda c: c["name"])
def test_build_context(case) -> None:
    index = VectorIndex()
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
    index = VectorIndex()
    chat = _make_chat(index)
    result = chat._build_context("test query")

    assert result == ""
