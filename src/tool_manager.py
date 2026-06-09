import json
import logging
from mcp_client import MCPClient
from mcp.types import CallToolResult, TextContent
from anthropic.types import Message, ToolResultBlockParam
from typing import Any, Optional, Literal

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
    ) -> Optional[MCPClient]:
        for client in clients:
            tools = await client.list_tools()
            tool = next((t for t in tools if t.name == tool_name), None)
            if tool:
                return client
        return None

    @staticmethod
    def _build_tool_result_part(
        tool_use_id: str, text: str, status: Literal["success", "error"]
    ) -> ToolResultBlockParam:
        return {
            "tool_use_id": tool_use_id,
            "type": "tool_result",
            "content": text,
            "is_error": status == "error",
        }

    @classmethod
    async def execute_tool_requests(
        cls, clients: dict[str, MCPClient], message: Message
    ) -> list[ToolResultBlockParam]:
        tool_requests = [block for block in message.content if block.type == "tool_use"]

        tool_result_blocks: list[ToolResultBlockParam] = []
        for tool_request in tool_requests:
            tool_use_id = tool_request.id
            tool_name = tool_request.name
            tool_input = tool_request.input

            client = await cls._find_client_with_tool(list(clients.values()), tool_name)

            if not client:
                tool_result_part = cls._build_tool_result_part(
                    tool_use_id=tool_use_id,
                    text="could not find the tools",
                    status="error",
                )

                tool_result_blocks.append(tool_result_part)
                continue

            try:
                tool_output: CallToolResult | None = await client.call_tool(
                    tool_name, tool_input
                )

                items = []
                if tool_output:
                    items = tool_output.content
                content_list = [
                    item.text for item in items if isinstance(item, TextContent)
                ]

                result_status: Literal["success", "error"] = (
                    "error" if tool_output and tool_output.isError else "success"
                )

                content_json = json.dumps(content_list)
                tool_result_part = cls._build_tool_result_part(
                    tool_use_id=tool_use_id,
                    text=content_json,
                    status=result_status,
                )
            except Exception as e:
                error_message = f"Error executing tool '{tool_name}': {e}"
                logger.error(error_message)
                tool_result_part = cls._build_tool_result_part(
                    tool_use_id=tool_use_id,
                    text=json.dumps({"error": error_message}),
                    status="error",
                )

            tool_result_blocks.append(tool_result_part)
        return tool_result_blocks
