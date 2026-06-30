build:
	docker build -t repo-lens .

run:
	docker run -it --env-file .env -v $(PWD)/src:/app/src -v /var/run/docker.sock:/var/run/docker.sock repo-lens

install:
	uv sync --extra dev

test:
	uv run pytest -v

e2e:
	uv run pytest e2e/ -v

test-all:
	uv run pytest tests/ e2e/ -v

lint:
	pre-commit run --all-files

eval:
	uv run src/prompt_eval.py "$(PROMPT)"

rag-eval:
	uv run src/rag_eval.py
