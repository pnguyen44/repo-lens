from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import CallToolResult, TextContent

from repo_lens.agents.tool_manager import ToolManager
from repo_lens.core.repo_context import RepoContext
from repo_lens.providers.chat_client import ToolCall


@pytest.mark.asyncio
async def test_execute_tool_requests_overrides_owner_and_repo_from_context() -> None:
    client = MagicMock()
    tool = MagicMock()
    tool.name = "get_file_contents"
    client.list_tools = AsyncMock(return_value=[tool])
    client.call_tool = AsyncMock(
        return_value=CallToolResult(
            content=[TextContent(type="text", text="ok")],
            isError=False,
        )
    )

    repo_context = RepoContext(owner="org", repo="my-repo")
    tool_calls = [
        ToolCall(
            id="call-1",
            name="get_file_contents",
            input={"owner": "wrong", "repo": "wrong", "path": "README.md"},
        )
    ]

    results = await ToolManager.execute_tool_requests(
        clients={"github": client},
        tool_calls=tool_calls,
        repo_context=repo_context,
    )

    client.call_tool.assert_called_once_with(
        "get_file_contents",
        {"owner": "org", "repo": "my-repo", "path": "README.md"},
    )
    assert len(results) == 1
    assert results[0]["is_error"] is False


@pytest.mark.asyncio
async def test_execute_tool_requests_without_context_keeps_tool_input() -> None:
    client = MagicMock()
    tool = MagicMock()
    tool.name = "get_file_contents"
    client.list_tools = AsyncMock(return_value=[tool])
    client.call_tool = AsyncMock(
        return_value=CallToolResult(
            content=[TextContent(type="text", text="ok")],
            isError=False,
        )
    )

    tool_input = {"owner": "org", "repo": "my-repo", "path": "README.md"}
    tool_calls = [ToolCall(id="call-1", name="get_file_contents", input=tool_input)]

    await ToolManager.execute_tool_requests(
        clients={"github": client},
        tool_calls=tool_calls,
        repo_context=None,
    )

    client.call_tool.assert_called_once_with("get_file_contents", tool_input)


@pytest.mark.asyncio
async def test_execute_tool_requests_calls_on_file_fetched_on_success() -> None:
    client = MagicMock()
    tool = MagicMock()
    tool.name = "get_file_contents"
    client.list_tools = AsyncMock(return_value=[tool])
    client.call_tool = AsyncMock(
        return_value=CallToolResult(
            content=[TextContent(type="text", text="ok")],
            isError=False,
        )
    )
    on_file_fetched = AsyncMock()

    tool_calls = [
        ToolCall(
            id="call-1",
            name="get_file_contents",
            input={"owner": "org", "repo": "my-repo", "path": "docs/foo.md"},
        )
    ]

    await ToolManager.execute_tool_requests(
        clients={"github": client},
        tool_calls=tool_calls,
        repo_context=None,
        on_file_fetched=on_file_fetched,
    )

    on_file_fetched.assert_called_once_with("docs/foo.md")


@pytest.mark.asyncio
async def test_execute_tool_requests_skips_on_file_fetched_when_tool_errors() -> None:
    client = MagicMock()
    tool = MagicMock()
    tool.name = "get_file_contents"
    client.list_tools = AsyncMock(return_value=[tool])
    client.call_tool = AsyncMock(
        return_value=CallToolResult(
            content=[TextContent(type="text", text="not found")],
            isError=True,
        )
    )
    on_file_fetched = AsyncMock()

    tool_calls = [
        ToolCall(
            id="call-1",
            name="get_file_contents",
            input={"owner": "org", "repo": "my-repo", "path": "docs/foo.md"},
        )
    ]

    await ToolManager.execute_tool_requests(
        clients={"github": client},
        tool_calls=tool_calls,
        repo_context=None,
        on_file_fetched=on_file_fetched,
    )

    on_file_fetched.assert_not_called()


@pytest.mark.asyncio
async def test_execute_tool_requests_skips_on_file_fetched_for_other_tools() -> None:
    client = MagicMock()
    tool = MagicMock()
    tool.name = "search_code"
    client.list_tools = AsyncMock(return_value=[tool])
    client.call_tool = AsyncMock(
        return_value=CallToolResult(
            content=[TextContent(type="text", text="ok")],
            isError=False,
        )
    )
    on_file_fetched = AsyncMock()

    tool_calls = [ToolCall(id="call-1", name="search_code", input={"query": "auth"})]

    await ToolManager.execute_tool_requests(
        clients={"github": client},
        tool_calls=tool_calls,
        repo_context=None,
        on_file_fetched=on_file_fetched,
    )

    on_file_fetched.assert_not_called()
