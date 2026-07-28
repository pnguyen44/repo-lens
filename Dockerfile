FROM python:3.11-slim

ARG INSTALL_DOCKER=true
RUN if [ "$INSTALL_DOCKER" = "true" ]; then \
        apt-get update && apt-get install -y docker.io && rm -rf /var/lib/apt/lists/*; \
    fi

ARG GITHUB_MCP_VERSION=1.7.0
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/* \
    && curl -L "https://github.com/github/github-mcp-server/releases/download/v${GITHUB_MCP_VERSION}/github-mcp-server_Linux_x86_64.tar.gz" \
    | tar xz -C /usr/local/bin github-mcp-server

WORKDIR /app

COPY . .

RUN pip install -e .

CMD ["python", "-u", "-m", "repo_lens.app.chat_cli"]
