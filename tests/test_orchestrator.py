from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from repo_lens.agents.agent import Agent, AgentName
from repo_lens.providers.chat_client import ChatResponse, StreamChunk, ToolCall
from repo_lens.agents.orchestrator import Orchestrator


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
async def test_run_calls_on_text_per_chunk() -> None:
    response = ChatResponse(stop_reason="end_turn", text="ab", tool_calls=[])
    chunks = [
        StreamChunk(type="text", text="a"),
        StreamChunk(type="text", text="b"),
    ]

    chat_client = MagicMock()
    chat_client.chat_stream.return_value = _make_stream(
        response=response, chunks=chunks
    )

    received: list[str] = []

    def on_text(text: str) -> None:
        received.append(text)

    orchestrator = Orchestrator(
        agents={AgentName.GITHUB: _make_agent()}, chat_client=chat_client
    )

    result = await orchestrator.run("hello", on_text=on_text)
    assert result == "ab"
    assert received == ["a", "b"]


@pytest.mark.asyncio
async def test_run_returns_text_when_planner_does_not_delegate() -> None:
    response = ChatResponse(
        stop_reason="end_turn", text="I can help with that.", tool_calls=[]
    )

    chunks = [
        StreamChunk(type="text", text="I can help "),
        StreamChunk(type="text", text="with that."),
    ]

    chat_client = MagicMock()
    chat_client.chat_stream.return_value = _make_stream(
        response=response, chunks=chunks
    )

    mock_agent = _make_agent()
    agents: dict[AgentName, Any] = {AgentName.GITHUB: mock_agent}
    orchestrator = Orchestrator(agents=agents, chat_client=chat_client)

    result = await orchestrator.run("hello")
    assert result == "I can help with that."
    chat_client.chat_stream.assert_called_once()


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
    chat_client.chat_stream.side_effect = [
        _make_stream(tool_call_response),
        _make_stream(
            final_response,
            [StreamChunk(type="text", text="here are the open PRs...")],
        ),
    ]

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
    chat_client.chat_stream.return_value = _make_stream(tool_call_response)

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
    chat_client.chat_stream.side_effect = [
        _make_stream(tool_call_response),
        _make_stream(
            final_response,
            [StreamChunk(type="text", text="I could not find that agent.")],
        ),
    ]

    mock_agent = _make_agent()
    agents: dict[AgentName, Any] = {AgentName.GITHUB: mock_agent}
    orchestrator = Orchestrator(agents=agents, chat_client=chat_client)

    await orchestrator.run("test_query")
    mock_agent.run.assert_not_called()
