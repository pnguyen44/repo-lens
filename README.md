# repo-lens

AI-powered CLI for chatting about GitHub conversations.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the App](#running-the-app)
- [Pre-commit Setup](#pre-commit-setup)

## Prerequisites

- [Python 3.11+](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)
- [Docker](https://docs.docker.com/get-started/get-docker/) (for containerized usage)

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

Then edit `.env` with your values. See `.env.example` for all available variables.

## Running the App

```bash
make build
make run
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
