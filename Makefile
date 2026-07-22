.PHONY: build chroma run run-ui run-ui-docker down clean-chroma install test e2e test-all lint prompt-eval rag-eval index

build:
	docker compose build

chroma:
	docker compose up -d chroma

run:
	docker compose down
	docker compose up -d chroma
	docker compose run --rm app

run-ui:
	PYTHONPATH=src uv run chainlit run chat_ui.py --port 8001 -w

run-ui-docker:
	docker compose up --build chroma app-ui

down:
	docker compose down

clean-chroma:
	docker compose down
	sudo rm -rf ./data/chroma

install:
	uv sync --extra dev

test:
	uv run pytest tests/ -v

e2e:
	uv run pytest e2e/ -v

test-all:
	uv run pytest tests/ e2e/ -v

lint:
	pre-commit run --all-files

prompt-eval:
	uv run python -m repo_lens.evals.prompt_eval "$(PROMPT)"

rag-eval:
	uv run python -m repo_lens.evals.rag_eval

index:
	uv run python -m repo_lens.rag.indexer
