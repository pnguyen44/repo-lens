from typing import Any

from chat_client import ChatClient
from mcp_client import MCPClient
from tool_manager import ToolManager


class Chat:
    def __init__(
        self, chat_client: ChatClient[Any, Any], mcp_clients: dict[str, MCPClient]
    ) -> None:
        self.chat_client = chat_client
        self.mcp_clients = mcp_clients
        self.tools: list[Any] = []
        self.messages: list[Any] = []

    async def run(self, query: str) -> str:
        final_text_response = ""

        if not self.tools:
            self.tools = await ToolManager.get_all_tools(self.mcp_clients)

        self.chat_client.add_user_message(self.messages, query)

        while True:
            response = self.chat_client.chat(messages=self.messages, tools=self.tools)

            self.chat_client.add_assistant_message(self.messages, response)

            if response.stop_reason == "tool_use":
                print(self.chat_client.text_from_message(response))
                print("Tool calls not yet implemented")

                break
            else:
                final_text_response = self.chat_client.text_from_message(response)
                print(final_text_response)
                break
        return final_text_response
