# Architecture

## Table of Contents

- [Core Flow](#core-flow)
  - [Orchestration](#orchestration)
  - [MCP Integration](#mcp-integration)
  - [Tool Use Loop](#tool-use-loop)
  - [RAG Pipeline](#rag-pipeline)
- [Data Layer](#data-layer)
  - [Indexing Lifecycle](#indexing-lifecycle)
  - [Persistence](#persistence)
- [Cross-cutting Concepts](#cross-cutting-concepts)
  - [Multi-Provider Support](#multi-provider-support)
  - [Streaming](#streaming)
  - [Prompt Caching](#prompt-caching)
  - [Structured Output](#structured-output)
  - [Observability](#observability)
  - [Error Handling](#error-handling)
- [Quality](#quality)
  - [Evaluation Pipeline](#evaluation-pipeline)

## Core Flow

### Orchestration

#### Sequence Diagram

```mermaid
sequenceDiagram
    participant User
    participant Orchestrator
    participant Planner
    participant GitHubAgent
    participant RAGAgent

    User->>Orchestrator: query
    Orchestrator->>Planner: query

    loop until planner returns text (max N delegations)
        Planner->>Orchestrator: delegate_to_agent(agent, task)

        alt agent = "github"
            Orchestrator->>GitHubAgent: task
            GitHubAgent-->>Orchestrator: result
        else agent = "rag"
            Orchestrator->>RAGAgent: task
            RAGAgent-->>Orchestrator: result
        end

        Orchestrator->>Planner: tool_result(result)
    end

    Planner-->>Orchestrator: final text response
    Orchestrator-->>User: final text response
```

#### Component Responsibilities

| Component | Owns | Does NOT do |
| --- | --- | --- |
| **Orchestrator** | Outer loop, agent registry, delegation routing, loop protection | LLM calls, tool execution |
| **Planner** | Reasoning, task decomposition, synthesis | MCP tool calls, RAG retrieval |
| **GitHubAgent** | GitHub MCP tools, repo queries, on-demand file indexing | Planning, synthesis |
| **RAGAgent** | Hybrid retrieval, reranking, codebase Q&A | Planning, MCP tool calls |

The active repository is passed as typed data through the orchestrator, automatically scoping prompts and tool calls to that repo.

#### Fail-Closed on Empty Delegation

When a specialist returns an empty result, the orchestrator retries once. If still empty, it returns an honest "I couldn't retrieve that information" message instead of forwarding the empty result to the planner.

Without this, the planner hallucinates from the empty tool result: it invents an answer rather than admitting it has no data. Observed in practice with Gemini (specialist produced no output, planner fabricated a response).

### MCP Integration

MCP (Model Context Protocol) servers run as subprocesses, connected via stdio. The system can register multiple MCP clients (e.g., GitHub).

On startup, tools are fetched from all connected servers. When the LLM requests a tool call, the system finds the right MCP client and dispatches the request. If a tool isn't found, an error is returned to the LLM so it can retry or respond gracefully.

### Tool Use Loop

Agents run an agentic loop: send a query to the LLM, check if it wants to call tools, execute the tools, feed results back, repeat until the LLM returns a text response.

The loop exits when:

- The LLM responds with text (no tool calls)
- Max tool iterations is reached (prevents runaway loops)
- An unrecoverable error occurs

This lets agents gather information across multiple tool calls before synthesizing a final answer.

### RAG Pipeline

Query flow: chunk → embed → hybrid search (vector + BM25 via RRF) → rerank → top-k context.

**Design decisions:**

- **Section-based chunking**. Content is split on `##` markdown headers, preserving natural topic boundaries. This fits the input (GitHub READMEs) better than fixed-size or sentence-based chunking, which would split mid-section.
- **Hybrid search** (vector + BM25). Vector search handles semantic similarity (paraphrased queries, synonyms) but misses exact terms. BM25 handles exact keyword matches but has no semantic understanding. Combining both with RRF surfaces documents that rank well across both systems.
- **Reranking**. Embedding search scores the query and each document independently. A cross-encoder reranker reads them together, producing more accurate relevance scores. It's too slow to run on the full index, so it only scores the top candidates from hybrid search.
- **K=3** (default). The K-sweep shows recall hits 100% at K=2 while precision drops with each additional chunk. K=3 adds a safety margin for recall. Retrieving 2-3 chunks gives the LLM richer context for synthesizing answers, even when not all chunks match the single expected section in the eval dataset.
- **Cosine similarity** (default). VoyageAI embeddings are unit-normalized (length 1), which makes cosine, dot product, and Euclidean distance rank-equivalent. All three produce identical retrieval results. Cosine is the conventional default across vector DBs and embedding providers.

## Data Layer

### Indexing Lifecycle

Files are indexed lazily, not eagerly. Eager indexing (embedding the entire repo upfront) wastes compute and API calls on files that are never queried.

- **On-demand indexing**: `GitHubAgent` indexes a file the first time the LLM fetches it via MCP. Each chunk is tagged with a `file_key`, so already-indexed files are skipped on refetch.
- **`/clear-cache`**: `DocumentIndexer` drops all chunks for the active repo from the vector store. The next file fetch re-indexes from scratch.
- **Startup sync**: `DocumentIndexer` rebuilds the in-memory BM25 index from the vector store on boot, so restarts don't re-embed (and re-bill) existing content.

### Persistence

| Layer | Backend | Lifetime |
| --- | --- | --- |
| Vector index | `QdrantVectorIndex` (default) or `ChromaVectorIndex`, selected via `VECTOR_STORE` config | Persistent; source of truth for indexed chunks |
| Keyword index | `BM25Index` | In-memory only; rebuilt from the vector store on startup |

Both vector backends implement `BaseVectorIndex`, so `HybridRetriever` and `DocumentIndexer` are backend-agnostic.

## Cross-cutting Concepts

### Multi-Provider Support

| Piece | Role |
| --- | --- |
| `ChatClientProtocol` | Common interface for chat, streaming, and message formatting across providers |
| `Claude` / `Gemini` | Provider-specific implementations (Anthropic SDK / Google GenAI SDK) |
| `create_chat_client` | Factory that picks a provider from `Config.provider` (`anthropic` default, or `gemini`) |
| `MessageStream` | Per-provider streaming protocol so the orchestrator loop stays provider-agnostic |

### Streaming

Responses stream token-by-token from the LLM to the caller (CLI or Chainlit).

Claude and Gemini SDKs send stream data in different formats. Each provider normalizes these into a common chunk format, so the rest of the app doesn't need to know which LLM is running.

The caller provides a callback that handles each text chunk as it arrives.

### Prompt Caching

System prompts and tool definitions are cached across requests using Anthropic's `cache_control`. Two `"ephemeral"` markers are placed: one on the system prompt block and one on the last tool definition. In a multi-turn conversation, the system prompt and tool schemas are identical every turn, so caching avoids re-processing them.

### Structured Output

LLM responses are parsed into Pydantic models (`model_validate_json`) to enforce schema compliance. When validation fails, the error message is fed back to the LLM for self-correction (retry with the validation error as context). This catches malformed JSON, missing fields, and wrong types instead of silent failures downstream.

Used in both eval pipelines to parse grading results from the LLM.

### Observability

Every query carries a `query_id` (8-char UUID) so logs can be traced end-to-end across orchestrator, agents, and providers. Logs are structured via structlog.

Key metrics tracked per request:

- `input_tokens`, `output_tokens`: cost visibility
- `cache_read_input_tokens`, `cache_creation_input_tokens`: prompt caching effectiveness

### Error Handling

Rate limits (429) are retried with exponential backoff, respecting the provider's retry-after when present.

- Max retries: 3
- Backoff: `2^retries` seconds (or provider's retry-after)
- On exhaustion: user-friendly message instead of crash

## Quality

### Evaluation Pipeline

Two eval pipelines measure system quality: RAG eval (retrieval + answer quality) and Prompt eval (routing correctness).

#### RAG Eval

Measures retrieval quality and answer faithfulness against a golden dataset.

| Metric | What it measures | How |
| --- | --- | --- |
| Section Precision | % of retrieved chunks that were expected | `retrieved ∩ expected / retrieved` |
| Section Recall | % of expected chunks that were retrieved | `retrieved ∩ expected / expected` |
| Keyword Recall | % of expected keywords found in retrieved content | Substring match |
| Faithfulness | Is the answer grounded in the context? | LLM-as-judge: `grounded / partial / hallucinated` |

Flow: load fixture → chunk & embed → run eval cases → compute metrics → print summary.

#### Prompt Eval

Measures whether prompts produce correct routing decisions (e.g., planner delegates to the right agent).

| Stage | What happens |
| --- | --- |
| Run prompt | Send prompt + test input to LLM with tools wired |
| Extract routing | Check which agent (if any) was called via `tool_calls` |
| Grade | Compare `actual` agent to `expected_agent` from test case |

Deterministic grading: pass if `actual == expected`, fail otherwise. No LLM judgment needed since it's an objective check.

#### Shared Patterns

Cross-cutting concerns that both eval pipelines share.

- **Structured output**: LLM responses are parsed into Pydantic models with self-correction on validation failure
- **Rate limit handling**: retry loop with exponential backoff for Anthropic/Gemini rate limits
- **Golden datasets**: hand-curated test cases in `*_eval_dataset.py` files
