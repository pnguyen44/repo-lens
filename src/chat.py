import asyncio
import logging
from typing import Any, Optional


from chat_client import ChatClient
from embeddings import Embedder
from hybrid_retriever import HybridRetriever
from mcp_client import MCPClient
from tool_manager import ToolManager
from anthropic import AuthenticationError, BadRequestError, RateLimitError
from reranker import Reranker

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


class Chat:
    def __init__(
        self,
        chat_client: ChatClient[Any, Any],
        mcp_clients: dict[str, MCPClient],
        system_prompt: str | None = None,
        embedder: Optional[Embedder] = None,
        hybrid_retriever: Optional[HybridRetriever] = None,
        web_search: bool = True,
        reranker: Optional[Reranker] = None,
    ) -> None:
        self.chat_client = chat_client
        self.mcp_clients = mcp_clients
        self.system_prompt = system_prompt
        self.tools: list[Any] = []
        self.messages: list[Any] = []
        self.embedder = embedder
        self.hybrid_retriever = hybrid_retriever
        self.web_search = web_search
        self.reranker = reranker

    def _build_context(self, query: str) -> str | list[Any]:
        if not self.hybrid_retriever:
            return ""

        try:
            results = self.hybrid_retriever.search(
                query_text=query, k=15 if self.reranker else 3
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
                chunks = [chunk for chunk, dist in results]

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

    async def run(self, query: str, tool_choice: dict[str, Any] | None = None) -> str:
        final_text_response = ""

        if not self.tools:
            self.tools = await ToolManager.get_all_tools(self.mcp_clients)

        augmented_query = self._build_context(query) or query
        self.chat_client.add_user_message(self.messages, augmented_query)

        retries = 0
        while True:
            try:
                with self.chat_client.chat_stream(
                    messages=self.messages,
                    tools=self.tools,
                    system=self.system_prompt,
                    web_search=self.web_search,
                    tool_choice=tool_choice,
                ) as stream:
                    current_block_type = None
                    for chunk in stream:
                        if chunk.type == "text":
                            print(chunk.text, end="")
                            final_text_response += chunk.text

                        if chunk.type == "content_block_start":
                            current_block_type = chunk.content_block.type
                            if chunk.content_block.type == "tool_use":
                                print(f'\n>>> Tool Call: "{chunk.content_block.name}"')

                        if chunk.type == "input_json" and chunk.partial_json:
                            print(chunk.partial_json, end="")

                        if chunk.type == "content_block_stop":
                            if current_block_type == "tool_use":
                                print()

                    response = stream.get_final_message()
                    self.chat_client.record_usage(response.usage)

                    if any(
                        b.type == "web_search_tool_result" for b in response.content
                    ):
                        logger.info("Web search tool called")

                    titles = self.chat_client.extract_citation_titles(response)
                    if titles:
                        print("\nSources: " + ", ".join(titles))

                    self.chat_client.add_assistant_message(self.messages, response)

                    if response.stop_reason != "tool_use":
                        break

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

                    if tool_choice:
                        tool_choice = None
                        continue

            except BadRequestError as e:
                if "credit balance is too low" in str(e):
                    print("\nOut of API credits. Switch provider or add credits.")
                    break
                if "prompt is too long" in str(e):
                    logger.warning("Conversation is too long. Starting fresh.")
                    self.messages.clear()
                    break
                raise
            except AuthenticationError:
                print("\nInvalid API key. Check your .env file.")
                break
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
