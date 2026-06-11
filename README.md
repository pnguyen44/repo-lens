# repo-lens

AI-powered CLI for chatting about GitHub repositories. Uses RAG (Retrieval-Augmented Generation) to index repo content and answer questions with relevant context.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the App](#running-the-app)
- [RAG Pipeline](#rag-pipeline)
- [RAG Evaluation](#rag-evaluation)
- [Prompt Eval Tool](#prompt-eval-tool)
- [Pre-commit Setup](#pre-commit-setup)

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

Then edit `.env` with your values:

- `ANTHROPIC_API_KEY` — Claude API key
- `CLAUDE_MODEL` — model to use (e.g. `claude-sonnet-4-20250514`)
- `GITHUB_TOKEN` — GitHub personal access token
- `DEFAULT_ORG` — default GitHub org for repo lookups
- `VOYAGE_API_KEY` — VoyageAI API key for embeddings

## Running the App

```bash
make build
make run
```

## RAG Pipeline

Indexes repository READMEs and uses vector search to provide relevant context to Claude when answering questions. Fetches content via GitHub MCP, chunks it, embeds with VoyageAI, and stores in an in-memory vector index.

## RAG Evaluation

Measures retrieval quality (precision, recall) against a golden eval dataset to verify the right content is being retrieved.

```bash
make rag-eval
```

## Prompt Eval Tool

Evaluate how well a prompt performs by auto-generating test cases and scoring responses with LLM-as-judge.

```bash
make eval PROMPT="Review the following code. Identify bugs and suggest improvements."
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
