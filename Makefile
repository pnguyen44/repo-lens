build:
	docker build -t repo-lens .

run:
	docker run -t --env-file .env -v $(PWD)/src:/app/src repo-lens

install:
	uv sync --extra dev

test:
	uv run pytest -v

lint:
	pre-commit run --all-files
