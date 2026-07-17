FROM python:3.11-slim

# Install Docker CLI so the container can spawn MCP server containers on the host
RUN apt-get update && apt-get install -y docker.io && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

# Editable install so the ./src volume mount picks up live code changes
RUN pip install -e .

CMD ["python", "-u", "-m", "repo_lens.app.chat_cli"]
