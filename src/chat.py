import logging
from typing import Any, Optional

from chat_client import ChatClient
from embeddings import Embedder
from mcp_client import MCPClient
from tool_manager import ToolManager
from anthropic import BadRequestError
from vector_index import VectorIndex

logger = logging.getLogger(__name__)

# Cosine distance (1 − similarity): 0.0 = identical, 2.0 = opposite.
# Chunks with distance <= this value are included as context in _build_context.
DISTANCE_THRESHOLD = 0.6


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

        for chunk, dist in results:
            logger.debug("Vector search distance: %s", dist)
            if dist <= DISTANCE_THRESHOLD:
                sources.append(
                    f'<source repo="{chunk["repo"]}" section="{chunk["section"]}">\n'
                    f"{chunk['content']}\n"
                    f"</source>"
                )

        if not sources:
            context = "No relevant source found"
        else:
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
                    logger.warning("Conversation is too long. Starting fresh.")
                    self.messages.clear()
                    break
                raise
        return final_text_response
