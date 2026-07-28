# RepoLens

An AI-powered tool for exploring and understanding GitHub repositories.

## What it can do

- **Hybrid RAG search** — combines vector similarity (VoyageAI embeddings) and BM25 keyword matching with Reciprocal Rank Fusion, then reranks results before answering
- **Query GitHub** — fetch issues, PRs, repo metadata via GitHub MCP tools through natural language
- **Multi-agent orchestration** — a planner–delegate system routes queries to specialist agents (RAG, GitHub) and synthesizes their results into a single response
