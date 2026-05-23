build:
	docker build -t repo-lens .

run:
	docker run -t --env-file .env -v $(PWD)/src:/app/src repo-lens

install:
	uv pip install -e .

lint:
	pre-commit run --all-files
