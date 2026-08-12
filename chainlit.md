# RepoLens

An AI-powered tool for exploring and understanding GitHub repositories.

## What it can do

- **Hybrid RAG search** — combines vector similarity (VoyageAI embeddings) and BM25 keyword matching with Reciprocal Rank Fusion, then reranks results before answering
- **Query GitHub** — fetch issues, PRs, repo metadata via GitHub MCP tools through natural language
- **Multi-agent orchestration** — a planner–delegate system routes queries to specialist agents (RAG, GitHub) and synthesizes their results into a single response

## Commands

- `/clear-cache` — clear the indexed knowledge for the active repo. Files are re-indexed automatically the next time they're fetched.
- `/org <name>` — set the active GitHub org for short `/repo` switches.
  - **Syntax:** `/org <name>`
  - **Example:** `/org openshift-hyperfleet`
  - **Notes:** org names are case-sensitive; no slashes.
- `/repo owner/repo` — switch the active repository mid-chat.
  - **Syntax:** `/repo owner/repo` or `/repo <repo-name>` after `/org <name>`
  - **Examples:** `/repo openshift-hyperfleet/hyperfleet-api` or `/org openshift-hyperfleet` then `/repo hyperfleet-api`
  - **Notes:** owner and repo names are case-sensitive; only public repositories are accessible.
