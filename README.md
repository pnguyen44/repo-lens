# repo-lens

AI-powered CLI and Chainlit UI for chatting about GitHub repositories. Uses RAG (Retrieval-Augmented Generation) to index repo content and answer questions with relevant context.

## Table of Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the App](#running-the-app)
- [Commands](#commands)
- [Chainlit UI](#chainlit-ui)
- [RAG Pipeline](#rag-pipeline)
- [RAG Evaluation](#rag-evaluation)
- [Prompt Eval Tool](#prompt-eval-tool)
- [Testing](#testing)
- [Pre-commit Setup](#pre-commit-setup)

## Architecture

A planner–delegate orchestrator routes each query to specialist agents (GitHub MCP tools and RAG over indexed repo content). See [ARCHITECTURE.md](ARCHITECTURE.md) for the sequence diagram, RAG pipeline, indexing lifecycle, and component responsibilities.

## Prerequisites

- [Python 3.11+](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)
- [Docker](https://docs.docker.com/get-started/get-docker/) (for containerized usage)
- [VoyageAI API key](https://www.voyageai.com/) (for embeddings)

## Installation

Create a virtual environment and install all dependencies:

```bash
uv venv
source .venv/bin/activate
make install
```

## Configuration

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

At minimum, you need:

- `MODEL` — LLM model to use (e.g. `claude-haiku-4-5`, `gemini-2.5-flash`)
- `GITHUB_TOKEN` — GitHub personal access token
- An API key for your chosen provider (`ANTHROPIC_API_KEY` or `GEMINI_API_KEY`)
- `VOYAGE_API_KEY` — VoyageAI API key for embeddings
- `DEFAULT_ORG` — GitHub org/owner for the startup repo
- `DEFAULT_REPO` — repository name

See `.env.example` for all available options and defaults.

For the Chainlit UI, also set:

- `APP_USER` — username for the login page
- `APP_PASS` — password for the login page
- `CHAINLIT_AUTH_SECRET` — secret for signing auth tokens (generate with `openssl rand -base64 32`)

Vector store configuration:

- `VECTOR_STORE` — `qdrant` (default) or `chroma`
- `QDRANT_URL` — Qdrant endpoint (e.g. `http://localhost:6333` for local, or your Qdrant Cloud URL)
- `QDRANT_API_KEY` — Qdrant Cloud API key (empty for local)

## Running the App

The app runs as a multi-container stack via Docker Compose: the app connects to Qdrant (vector search) and ChromaDB (legacy fallback) for persistent storage. Set `VECTOR_STORE=qdrant` (default) or `VECTOR_STORE=chroma` to choose the backend.

```bash
make run-cli
```

This starts the ChromaDB server, waits for it to be healthy, and launches the CLI. Source changes are picked up automatically via volume mount. Run `make build` after changing dependencies.

To stop all containers:

```bash
make down
```

## Commands

Available in both the CLI and Chainlit UI:

- `/clear-cache` — clear indexed knowledge for the active repo
- `/repo owner/repo` — switch the active repository
- `quit` / `exit` — leave the CLI

See [chainlit.md](chainlit.md) for syntax, examples, and notes.

## Chainlit UI

A web chat UI over the same `App` / orchestrator stack as the CLI. Entry point is `chat_ui.py`. Protected by username/password login (set `APP_USER` and `APP_PASS` in `.env`). See [chainlit.md](chainlit.md) for the welcome screen users see on first load.

Qdrant uses port `6333`; Chainlit uses `8001` so they do not clash.

**Local (uv):**

```bash
make chroma
make run-ui
```

Open [http://localhost:8001](http://localhost:8001).

**Docker (same image as the CLI, different command):**

```bash
make run-ui-docker
```

Then open [http://localhost:8001](http://localhost:8001). Stop with `make down`.

## Deployment

Deployed on [Render](https://render.com) (free tier) with [Qdrant Cloud](https://cloud.qdrant.io) for persistent vector storage.

**To deploy your own instance:**

1. Fork this repo
2. Create a free Qdrant Cloud cluster
3. On Render, create a new Blueprint and connect the repo
4. Fill in the secrets in the Render dashboard (API keys, auth credentials)
5. Deploy — the app indexes the repo on first boot

See `render.yaml` for the full service configuration.

## RAG Pipeline

Indexes repository READMEs and uses hybrid search (vector + BM25) to provide relevant context when answering questions. Fetches content via GitHub MCP, chunks it, embeds with VoyageAI, and stores in a vector index.

```bash
make index
```

Files are also indexed on demand as the LLM fetches them during a conversation.

## RAG Evaluation

Measures retrieval quality (precision, recall) against a golden eval dataset to verify the right content is being retrieved.

```bash
make rag-eval
```

## Prompt Eval Tool

Evaluate how well a prompt performs by auto-generating test cases and scoring responses with LLM-as-judge.

```bash
make prompt-eval PROMPT="Review the following code. Identify bugs and suggest improvements."
```

## Testing

Run unit tests:

```bash
make test
```

Run end-to-end tests (requires API keys configured in `.env`):

```bash
make e2e
```

Run all tests:

```bash
make test-all
```

## Pre-commit Setup

This project uses [pre-commit](https://pre-commit.com/) to run linting and formatting checks before each commit.

### Install pre-commit

```bash
uv pip install pre-commit
```

### Install the git hooks

```bash
pre-commit install
```

### Run hooks manually (optional)

To run all hooks against every file:

```bash
pre-commit run --all-files
```
