from mcp_client import MCPClient
from mcp.types import Tool


class ToolManager:
    @classmethod
    async def get_all_tools(cls, clients: dict[str, MCPClient]) -> list[Tool]:
        tools = []
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
