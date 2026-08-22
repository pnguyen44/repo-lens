.PHONY: build chroma run-cli run-ui run-ui-docker down clean-chroma install test e2e test-all lint prompt-eval rag-eval rag-eval-sweep index

build:
	docker compose build

chroma:
	docker compose up -d chroma

run-cli:
	docker compose down
	docker compose up -d chroma
	docker compose run --rm app

run-ui:
	@echo "Open http://localhost:8001"
	docker compose up -d qdrant chroma
	PYTHONPATH=src uv run chainlit run chat_ui.py --port 8001 --headless

run-ui-docker:
	@echo "Open http://localhost:8001"
	docker compose up --build qdrant chroma app-ui

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
	uv run python -m repo_lens.evals.prompt.prompt_eval $(if $(PROMPT),"$(PROMPT)")

rag-eval:
	uv run python -m repo_lens.evals.rag.rag_eval

rag-eval-sweep:
	uv run python -m repo_lens.evals.rag.rag_eval --sweep

index:
	uv run python -m repo_lens.rag.indexer
