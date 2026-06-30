FROM python:3.11-slim

# Install Docker CLI so the container can spawn MCP server containers on the host
RUN apt-get update && apt-get install -y docker.io && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install .

WORKDIR /app/src

CMD ["python", "-u", "chat_cli.py"]
