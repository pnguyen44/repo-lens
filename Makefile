build:
	docker build -t repo-lens .

run:
	docker run -it --env-file .env -v $(PWD)/src:/app/src -v /var/run/docker.sock:/var/run/docker.sock repo-lens

install:
	uv sync --extra dev

test:
	uv run pytest -v

lint:
	pre-commit run --all-files
