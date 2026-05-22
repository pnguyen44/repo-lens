build:
	docker build -t repo-lens .

run:
	docker run --env-file .env -v $(PWD)/src:/app/src repo-lens

lint:
	pre-commit run --all-files
