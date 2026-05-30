from typing import Any

from chat_client import ChatClient
from mcp_client import MCPClient
from tool_manager import ToolManager
from anthropic import BadRequestError


class Chat:
    def __init__(
        self,
        chat_client: ChatClient[Any, Any],
        mcp_clients: dict[str, MCPClient],
        system_prompt: str | None = None,
    ) -> None:
        self.chat_client = chat_client
        self.mcp_clients = mcp_clients
        self.system_prompt = system_prompt
        self.tools: list[Any] = []
        self.messages: list[Any] = []

    async def run(self, query: str) -> str:
        final_text_response = ""

        if not self.tools:
            self.tools = await ToolManager.get_all_tools(self.mcp_clients)

        self.chat_client.add_user_message(self.messages, query)

        while True:
            try:
                response = self.chat_client.chat_stream(
                    messages=self.messages, tools=self.tools, system=self.system_prompt
                )

                self.chat_client.add_assistant_message(self.messages, response)

                if response.stop_reason == "tool_use":
                    tool_result_parts = await ToolManager.execute_tool_requests(
                        clients=self.mcp_clients, message=response
                    )

                    self.chat_client.add_user_message(
                        messages=self.messages,
                        content=tool_result_parts,
                    )
                else:
                    final_text_response = self.chat_client.text_from_message(response)
                    break
            except BadRequestError as e:
                if "prompt is too long" in str(e):
                    print("\n Conversation is too long. Starting fresh.\n")
                    self.messages.clear()
                    break
                raise
        return final_text_response
