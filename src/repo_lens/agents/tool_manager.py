import json
import logging
from typing import Any, Awaitable, Callable, Literal

from mcp.types import CallToolResult, TextContent

from repo_lens.core.mcp_client import MCPClient
from repo_lens.core.repo_context import RepoContext
from repo_lens.providers.chat_client import ToolCall

logger = logging.getLogger(__name__)


class ToolManager:
    @classmethod
    async def get_all_tools(cls, clients: dict[str, MCPClient]) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for client in clients.values():
            tool_models = await client.list_tools()
            tools += [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.inputSchema,
                }
                for t in tool_models
            ]
        return tools

    @classmethod
    async def _find_client_with_tool(
        cls, clients: list[MCPClient], tool_name: str
    ) -> MCPClient | None:
        for client in clients:
            tools = await client.list_tools()
            tool = next((t for t in tools if t.name == tool_name), None)
            if tool:
                return client
        return None

    @staticmethod
    def _build_tool_result_part(
        tool_use_id: str, name: str, text: str, status: Literal["success", "error"]
    ) -> dict[str, Any]:
        return {
            "tool_use_id": tool_use_id,
            "name": name,
            "type": "tool_result",
            "content": text,
            "is_error": status == "error",
        }

    @classmethod
    async def execute_tool_requests(
        cls,
        clients: dict[str, MCPClient],
        tool_calls: list[ToolCall],
        repo_context: RepoContext | None = None,
        on_file_fetched: Callable[[str], Awaitable[None]] | None = None,
    ) -> list[dict[str, Any]]:
        tool_result_blocks: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            tool_use_id = tool_call.id
            tool_name = tool_call.name
            tool_input = tool_call.input

            if repo_context is not None:
                tool_input = {
                    **tool_input,
                    "owner": repo_context.owner,
                    "repo": repo_context.repo,
                }

            client = await cls._find_client_with_tool(list(clients.values()), tool_name)

            if not client:
                tool_result_part = cls._build_tool_result_part(
                    tool_use_id=tool_use_id,
                    name=tool_name,
                    text="could not find the tools",
                    status="error",
                )

                tool_result_blocks.append(tool_result_part)
                continue

            try:
                tool_output: CallToolResult | None = await client.call_tool(
                    tool_name, tool_input
                )
                logger.debug("tool call: %s(%s)", tool_name, tool_input)

                items = []
                if tool_output:
                    items = tool_output.content
                content_list = [
                    item.text for item in items if isinstance(item, TextContent)
                ]
                logger.debug("tool result: %s", content_list)

                result_status: Literal["success", "error"] = (
                    "error" if tool_output and tool_output.isError else "success"
                )

                content_json = json.dumps(content_list)

                if (
                    on_file_fetched
                    and result_status == "success"
                    and tool_name == "get_file_contents"
                ):
                    fetch_path = tool_input.get("path", "")
                    if isinstance(fetch_path, str):
                        await on_file_fetched(fetch_path)

                tool_result_part = cls._build_tool_result_part(
                    tool_use_id=tool_use_id,
                    name=tool_name,
                    text=content_json,
                    status=result_status,
                )
            except Exception as e:
                error_message = f"Error executing tool '{tool_name}': {e}"
                logger.error(error_message)
                tool_result_part = cls._build_tool_result_part(
                    tool_use_id=tool_use_id,
                    name=tool_name,
                    text=json.dumps({"error": error_message}),
                    status="error",
                )

            tool_result_blocks.append(tool_result_part)
        return tool_result_blocks
