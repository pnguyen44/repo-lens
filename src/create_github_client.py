from mcp_client import MCPClient


def create_github_client(github_token: str) -> MCPClient:
    return MCPClient(
        command="docker",
        args=[
            "run",
            "-i",
            "--rm",
            "-e",
            "GITHUB_PERSONAL_ACCESS_TOKEN",
            "ghcr.io/github/github-mcp-server",
        ],
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": github_token},
    )
