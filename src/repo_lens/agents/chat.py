import asyncio
import logging
import time
from enum import Enum
from typing import Any, Protocol

from anthropic import AuthenticationError, BadRequestError
from anthropic import RateLimitError as AnthropicRateLimitError
from google.genai.errors import ClientError as GeminiClientError

from repo_lens.agents.tool_manager import ToolManager
from repo_lens.core.config import DEFAULT_MAX_TOOL_ITERATIONS
from repo_lens.core.mcp_client import MCPClient
from repo_lens.core.repo_context import RepoContext
from repo_lens.core.retry import wait_for_retry
from repo_lens.core.trace_context import start_query_trace
from repo_lens.providers.chat_client import ChatClientProtocol, StreamError
from repo_lens.rag.embeddings import Embedder
from repo_lens.rag.hybrid_retriever import HybridRetriever
from repo_lens.rag.reranker import Reranker

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


class OnToolStartCallback(Protocol):
    def __call__(self, tool_name: str) -> None: ...


class OnToolInputCallback(Protocol):
    def __call__(self, partial_json: str) -> None: ...


class OnFileFetchedCallback(Protocol):
    async def __call__(self, path: str) -> None: ...


class RunStatus(Enum):
    DONE = "done"
    TOOL_USE = "tool_use"
    ERROR = "error"


class Chat:
    def __init__(
        self,
        *,
        chat_client: ChatClientProtocol,
        mcp_clients: dict[str, MCPClient] | None = None,
        system_prompt: str | None = None,
        embedder: Embedder | None = None,
        hybrid_retriever: HybridRetriever | None = None,
        web_search: bool = True,
        reranker: Reranker | None = None,
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
    ) -> None:
        self.chat_client = chat_client
        self.mcp_clients = mcp_clients or {}
        self.system_prompt = system_prompt
        self.tools: list[Any] = []
        self.messages: list[Any] = []
        self.embedder = embedder
        self.hybrid_retriever = hybrid_retriever
        self.web_search = web_search
        self.reranker = reranker
        self.repo_context: RepoContext | None = None
        self._stream_had_output = False
        self.max_tool_iterations = max_tool_iterations

    def _build_context(self, query: str) -> str | list[Any]:
        if not self.hybrid_retriever:
            return ""

        try:
            k = 15 if self.reranker else 3
            results = self.hybrid_retriever.search(
                query_text=query,
                k=k,
                repo=self.repo_context.key if self.repo_context else None,
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

    async def _prepare_query(self, query: str) -> None:
        if not self.tools:
            self.tools = await ToolManager.get_all_tools(self.mcp_clients)

        augmented_query = await asyncio.to_thread(self._build_context, query) or query
        self.chat_client.add_user_message(self.messages, augmented_query)

    def _stream_and_log(
        self,
        tool_choice: dict[str, Any] | None = None,
        on_tool_start: OnToolStartCallback | None = None,
        on_tool_input: OnToolInputCallback | None = None,
    ) -> tuple[Any, str]:
        call_start = time.perf_counter()
        system = (self.system_prompt or "") + (
            self.repo_context.prompt_suffix() if self.repo_context else ""
        )
        text = ""
        had_output = False
        try:
            with self.chat_client.chat_stream(
                messages=self.messages,
                tools=self.tools,
                system=system,
                web_search=self.web_search,
                tool_choice=tool_choice,
            ) as stream:
                for chunk in stream:
                    if chunk.type == "text":
                        text += chunk.text
                        had_output = True
                    elif chunk.type == "tool_start":
                        if on_tool_start:
                            on_tool_start(chunk.tool_name)
                    elif chunk.type == "tool_input":
                        if on_tool_input:
                            on_tool_input(chunk.partial_json)

                response = stream.get_final_message()

                call_ms = (time.perf_counter() - call_start) * 1000
                if response.usage:
                    self.chat_client.record_usage(response.usage)
                    usage = response.usage
                    in_tok = (
                        usage.get("input_tokens", 0)
                        if isinstance(usage, dict)
                        else getattr(usage, "input_tokens", 0)
                    )
                    out_tok = (
                        usage.get("output_tokens", 0)
                        if isinstance(usage, dict)
                        else getattr(usage, "output_tokens", 0)
                    )
                    logger.info(
                        "llm call",
                        extra={
                            "call_ms": round(call_ms, 2),
                            "input_tokens": in_tok,
                            "output_tokens": out_tok,
                        },
                    )

                return response, text
        finally:
            self._stream_had_output = had_output

    def _process_response(self, response: Any) -> RunStatus:
        if not response.raw:
            logger.warning(
                "Stream completed but no raw response available (stop_reason=%s)",
                response.stop_reason,
            )
            return RunStatus.ERROR
        if self.chat_client.has_web_search_results(response.raw):
            logger.debug("Web search tool called")

        titles = self.chat_client.extract_citation_titles(response.raw)
        if titles:
            logger.debug("Sources: %s", ", ".join(titles))

        self.chat_client.add_assistant_message(self.messages, response)

        if response.stop_reason != "tool_use":
            return RunStatus.DONE

        return RunStatus.TOOL_USE

    async def _execute_tools(
        self,
        response: Any,
        *,
        on_file_fetched: OnFileFetchedCallback | None = None,
    ) -> None:
        tool_names = [
            b.get("name") if isinstance(b, dict) else getattr(b, "name", None)
            for b in response.tool_calls
        ]
        logger.debug("Tool call: %s", tool_names)

        tool_result_parts = await ToolManager.execute_tool_requests(
            clients=self.mcp_clients,
            tool_calls=response.tool_calls,
            repo_context=self.repo_context,
            on_file_fetched=on_file_fetched,
        )

        self.chat_client.add_user_message(
            messages=self.messages,
            content=tool_result_parts,
        )

    async def run(
        self,
        query: str,
        repo_context: RepoContext | None = None,
        tool_choice: dict[str, Any] | None = None,
        on_tool_start: OnToolStartCallback | None = None,
        on_tool_input: OnToolInputCallback | None = None,
        on_file_fetched: OnFileFetchedCallback | None = None,
    ) -> str:
        start_query_trace()
        self.repo_context = repo_context
        start_query = time.perf_counter()
        final_text_response = ""

        await self._prepare_query(query)

        retries = 0
        tool_rounds = 0
        while True:
            try:
                response, text = self._stream_and_log(
                    tool_choice=tool_choice,
                    on_tool_start=on_tool_start,
                    on_tool_input=on_tool_input,
                )
                final_text_response += text

                status = self._process_response(response)
                if status != RunStatus.TOOL_USE:
                    break

                tool_rounds += 1
                if tool_rounds > self.max_tool_iterations:
                    logger.warning(
                        "Max tool iterations (%d) reached; stopping tool loop",
                        self.max_tool_iterations,
                    )
                    break

                await self._execute_tools(response, on_file_fetched=on_file_fetched)
                tool_choice = None

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
            except AnthropicRateLimitError:
                if self._stream_had_output:
                    logger.warning(
                        "Rate limited after partial stream output; not retrying."
                    )
                    break
                if await wait_for_retry(retries=retries, max_retries=MAX_RETRIES):
                    break
                retries += 1
                continue
            except GeminiClientError as e:
                if e.code != 429:
                    raise
                if self._stream_had_output:
                    logger.warning(
                        "Rate limited after partial stream output; not retrying."
                    )
                    break
                if await wait_for_retry(
                    retries=retries,
                    max_retries=MAX_RETRIES,
                    detail=getattr(e, "message", str(e)),
                ):
                    break
                retries += 1
                continue
            except StreamError as e:
                logger.error("Stream error: %s", e)
                print(f"\nStream error: {e}")
                break
            except Exception as e:
                logger.exception("Unexpected error: %s", e)
                print(f"\nUnexpected error: {e}")
                break
        duration_ms = (time.perf_counter() - start_query) * 1000
        logger.info("query completed", extra={"duration_ms": round(duration_ms, 2)})
        return final_text_response
