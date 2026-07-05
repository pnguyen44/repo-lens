.PHONY: build run install test e2e test-all lint prompt-eval rag-eval

build:
	docker build -t repo-lens .

run:
	docker run -it --env-file .env -v $(PWD)/src:/app/src -v /var/run/docker.sock:/var/run/docker.sock repo-lens

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
