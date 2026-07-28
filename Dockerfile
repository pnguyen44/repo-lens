FROM python:3.11-slim

ARG INSTALL_DOCKER=true
RUN if [ "$INSTALL_DOCKER" = "true" ]; then \
        apt-get update && apt-get install -y docker.io && rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /app

COPY . .

RUN pip install -e .

CMD ["python", "-u", "-m", "repo_lens.app.chat_cli"]
