.PHONY: build chroma run down clean-chroma install test e2e test-all lint prompt-eval rag-eval index

build:
	docker compose build

chroma:
	docker compose up -d chroma

run:
	docker compose down
	docker compose up -d chroma
	docker compose run --build --rm app

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
	uv run src/prompt_eval.py "$(PROMPT)"

rag-eval:
	uv run src/rag_eval.py

index:
	uv run src/indexer.py
