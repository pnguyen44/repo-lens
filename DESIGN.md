# Agent Orchestrator Design

## Sequence Diagram

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

## Component Responsibilities

| Component | Owns | Does NOT do |
|---|---|---|
| **Orchestrator** | Outer loop, agent registry, delegation routing, loop protection | LLM calls, tool execution |
| **Planner** | Reasoning, task decomposition, synthesis | MCP tool calls, RAG retrieval |
| **GitHubAgent** | GitHub MCP tools, repo queries | Planning, synthesis |
| **RAGAgent** | Hybrid retrieval, reranking, codebase Q&A | Planning, MCP tool calls |
