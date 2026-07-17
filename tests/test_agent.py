from unittest.mock import AsyncMock, MagicMock

import pytest

from repo_lens.agents.agent import (
    Agent,
    AgentName,
    create_github_agent,
    create_rag_agent,
)


def test_create_github_agent_wires_correct_config() -> None:
    chat_client = MagicMock()
    github_mcp = MagicMock()
    agent = create_github_agent(chat_client=chat_client, github_mcp=github_mcp)

    assert agent.name == AgentName.GITHUB
    assert agent.chat.mcp_clients == {"github": github_mcp}
    assert agent.chat.web_search is False


def test_create_rag_agent_wires_correct_config() -> None:
    chat_client = MagicMock()
    retriever = MagicMock()
    reranker = MagicMock()
    agent = create_rag_agent(
        chat_client=chat_client, hybrid_retriever=retriever, reranker=reranker
    )

    assert agent.name == AgentName.RAG
    assert agent.chat.hybrid_retriever is retriever
    assert agent.chat.reranker is reranker
    assert agent.chat.web_search is False


def test_create_rag_agent_without_reranker() -> None:
    chat_client = MagicMock()
    retriever = MagicMock()
    agent = create_rag_agent(chat_client=chat_client, hybrid_retriever=retriever)

    assert agent.chat.reranker is None


@pytest.mark.asyncio
async def test_agent_run_clears_messages_and_delegates() -> None:
    mock_chat = MagicMock()
    mock_chat.run = AsyncMock(return_value="answer")
    mock_chat.messages = [{"role": "user", "content": "old message"}]

    agent = Agent(name=AgentName.GITHUB, chat=mock_chat, description="test agent")
    result = await agent.run("some task")

    assert mock_chat.messages == []
    assert result == "answer"
    mock_chat.run.assert_called_once_with(
        query="some task", on_tool_start=None, on_tool_input=None
    )
