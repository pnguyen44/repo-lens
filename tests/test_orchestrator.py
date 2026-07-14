from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent import Agent, AgentName
from chat_client import ChatResponse, ToolCall
from orchestrator import Orchestrator


def _make_agent(run_result: str | None = None) -> MagicMock:
    mock_agent = MagicMock(spec=Agent)
    mock_agent.description = "test agent"
    if run_result is not None:
        mock_agent.run = AsyncMock(return_value=run_result)
    return mock_agent


def _make_delegation_call(agent_name: str, task: str, call_id: str = "1") -> ToolCall:
    return ToolCall(
        id=call_id,
        name="delegate_to_agent",
        input={"agent_name": agent_name, "task": task},
    )


@pytest.mark.asyncio
async def test_run_returns_text_when_planner_does_not_delegate() -> None:
    response = ChatResponse(
        stop_reason="end_turn", text="I can help with that.", tool_calls=[]
    )

    chat_client = MagicMock()
    chat_client.chat.return_value = response

    mock_agent = _make_agent()
    agents: dict[AgentName, Any] = {AgentName.GITHUB: mock_agent}
    orchestrator = Orchestrator(agents=agents, chat_client=chat_client)

    result = await orchestrator.run("hello")
    assert result == "I can help with that."
    chat_client.chat.assert_called_once()


@pytest.mark.asyncio
async def test_run_delegates_to_agent_and_returns_synthesized_response() -> None:
    tool_call_response = ChatResponse(
        stop_reason="tool_use",
        text="",
        tool_calls=[_make_delegation_call("github", "list ")],
    )

    final_response = ChatResponse(
        stop_reason="end_turn", text="here are the open PRs...", tool_calls=[]
    )

    chat_client = MagicMock()
    chat_client.chat.side_effect = [tool_call_response, final_response]

    mock_agent = _make_agent(run_result="3 open PRs found")
    agents: dict[AgentName, Any] = {AgentName.GITHUB: mock_agent}
    orchestrator = Orchestrator(agents=agents, chat_client=chat_client)

    result = await orchestrator.run("test_query")
    assert result == "here are the open PRs..."


@pytest.mark.asyncio
async def test_run_stops_at_max_delegations() -> None:
    tool_call_response = ChatResponse(
        stop_reason="tool_use",
        text="",
        tool_calls=[_make_delegation_call("github", "some task")],
    )

    chat_client = MagicMock()
    chat_client.chat.return_value = tool_call_response

    mock_agent = _make_agent(run_result="result")
    agents: dict[AgentName, Any] = {AgentName.GITHUB: mock_agent}
    orchestrator = Orchestrator(
        agents=agents, chat_client=chat_client, max_delegations=2
    )

    await orchestrator.run("test_query")
    assert mock_agent.run.call_count <= 2


@pytest.mark.asyncio
async def test_run_handles_unknown_agent_name() -> None:
    tool_call_response = ChatResponse(
        stop_reason="tool_use",
        text="",
        tool_calls=[_make_delegation_call("unknown", "some task")],
    )

    final_response = ChatResponse(
        stop_reason="end_turn", text="I could not find that agent.", tool_calls=[]
    )

    chat_client = MagicMock()
    chat_client.chat.side_effect = [tool_call_response, final_response]

    mock_agent = _make_agent()
    agents: dict[AgentName, Any] = {AgentName.GITHUB: mock_agent}
    orchestrator = Orchestrator(agents=agents, chat_client=chat_client)

    await orchestrator.run("test_query")
    mock_agent.run.assert_not_called()
