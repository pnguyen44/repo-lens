import asyncio
import logging
from typing import Any, Optional

from chat_client import ChatClient
from embeddings import Embedder, InputType
from mcp_client import MCPClient
from tool_manager import ToolManager
from anthropic import BadRequestError, RateLimitError
from vector_index import VectorIndex
from reranker import Reranker

logger = logging.getLogger(__name__)

# Cosine distance (1 − similarity): 0.0 = identical, 2.0 = opposite.
# Chunks with distance <= this value are included as context in _build_context.
DISTANCE_THRESHOLD = 0.6
MAX_RETRIES = 3


class Chat:
    def __init__(
        self,
        chat_client: ChatClient[Any, Any],
        mcp_clients: dict[str, MCPClient],
        system_prompt: str | None = None,
        embedder: Optional[Embedder] = None,
        index: Optional[VectorIndex] = None,
        web_search: bool = True,
        reranker: Optional[Reranker] = None,
    ) -> None:
        self.chat_client = chat_client
        self.mcp_clients = mcp_clients
        self.system_prompt = system_prompt
        self.tools: list[Any] = []
        self.messages: list[Any] = []
        self.embedder = embedder
        self.index = index
        self.web_search = web_search
        self.reranker = reranker

    def _build_context(self, query: str) -> str | list[Any]:
        if not self.embedder or not self.index:
            return ""

        try:
            query_vector = self.embedder.generate_embeddings(
                [query], input_type=InputType.QUERY
            )[0]
            results = self.index.search(
                query=query_vector, k=15 if self.reranker else 3
            )

            if self.reranker:
                docs = [doc["content"] for (doc, _dist) in results]
                reranked = self.reranker.rerank(query=query, documents=docs, top_k=3)
                chunks = []
                for r in reranked:
                    if 0 <= r.index < len(results):
                        chunks.append(results[r.index][0])
                    else:
                        logger.warning(
                            "Reranker returned out-of-bounds index: %d", r.index
                        )

            else:
                chunks = [
                    chunk for chunk, dist in results if dist <= DISTANCE_THRESHOLD
                ]

            sources = [
                self.chat_client.build_document_block(
                    content=c["content"], title=c["url"]
                )
                for c in chunks
            ]

            if not sources:
                return ""
            sources.append({"type": "text", "text": query})

            return sources
        except Exception:
            logger.warning("context retrieval failed, proceeding without RAG")
            return ""

    async def run(self, query: str) -> str:
        final_text_response = ""

        if not self.tools:
            self.tools = await ToolManager.get_all_tools(self.mcp_clients)

        augmented_query = self._build_context(query) or query
        self.chat_client.add_user_message(self.messages, augmented_query)

        retries = 0
        while True:
            try:
                response = self.chat_client.chat_stream(
                    messages=self.messages,
                    tools=self.tools,
                    system=self.system_prompt,
                    web_search=self.web_search,
                )

                self.chat_client.add_assistant_message(self.messages, response)

                if response.stop_reason == "tool_use":
                    tool_names = [
                        b.name for b in response.content if b.type == "tool_use"
                    ]
                    logger.info("Tool call: %s", tool_names)
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
            except RateLimitError:
                if retries >= MAX_RETRIES:
                    logger.error("Rate limited after %d retries. Skipping.", retries)
                    break
                wait = 2**retries
                logger.warning("Rate limited. Retrying in %ds...", wait)
                await asyncio.sleep(wait)
                retries += 1
                continue
        return final_text_response
