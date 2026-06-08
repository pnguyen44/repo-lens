from typing import Any

from chat_client import ChatClient
from embeddings import Embedder
from mcp_client import MCPClient
from tool_manager import ToolManager
from anthropic import BadRequestError
from typing import Optional
from vector_index import VectorIndex


class Chat:
    def __init__(
        self,
        chat_client: ChatClient[Any, Any],
        mcp_clients: dict[str, MCPClient],
        system_prompt: str | None = None,
        embedder: Optional[Embedder] = None,
        index: Optional[VectorIndex] = None,
    ) -> None:
        self.chat_client = chat_client
        self.mcp_clients = mcp_clients
        self.system_prompt = system_prompt
        self.tools: list[Any] = []
        self.messages: list[Any] = []
        self.embedder = embedder
        self.index = index

    def _build_context(self, query: str) -> str:
        if not self.embedder or not self.index:
            return ""
        query_vector = self.embedder.generate_embeddings([query])[0]
        results = self.index.search(query_vector, k=3)

        sources = []

        for chunk, _ in results:
            sources.append(
                f'<source repo="{chunk["repo"]}" section="{chunk["section"]}">\n'
                f"{chunk['content']}\n"
                f"</source>"
            )

        context = "\n".join(sources)

        augmented_query = f"""
        <context>
        {context}
        </context>

        {query}
        """

        return augmented_query

    async def run(self, query: str) -> str:
        final_text_response = ""

        if not self.tools:
            self.tools = await ToolManager.get_all_tools(self.mcp_clients)

        augmented_query = self._build_context(query) or query
        self.chat_client.add_user_message(self.messages, augmented_query)

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
