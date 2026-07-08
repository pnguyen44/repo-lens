import logging
import os
from types import TracebackType
from typing import Self

from mcp import ClientSession, StdioServerParameters, types
from contextlib import AsyncExitStack

from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class MCPClient:
    def __init__(
        self, command: str, args: list[str], env: dict[str, str] | None = None
    ) -> None:
        self._command = command
        self._args = args
        self._env = env
        self._session: ClientSession | None = None
        self._exit_stack = AsyncExitStack()

    async def connect(self) -> None:
        # Define how to start the MCP server (command, args, env vars)
        server_params = StdioServerParameters(
            command=self._command, args=self._args, env=self._env
        )

        # Spawn the server as a subprocess and open a stdio transport
        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )

        # Create a session over the transport for sending/receiving messages
        _stdio, _write = stdio_transport
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(_stdio, _write)
        )
        assert self._session is not None

        # Handshake with the server (capability negotiation)
        await self._session.initialize()

    def session(self) -> ClientSession:
        if self._session is None:
            raise ConnectionError("Client session not initialized.")
        return self._session

    async def list_tools(self) -> list[types.Tool]:
        result = await self.session().list_tools()
        return list(result.tools)

    async def call_tool(
        self, name: str, arguments: dict[str, object] | None = None
    ) -> types.CallToolResult:
        return await self.session().call_tool(name=name, arguments=arguments)

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def cleanup(self) -> None:
        try:
            await self._exit_stack.aclose()
        except Exception as e:
            logger.debug("MCP cleanup error (expected on exit): %s", e)

        self._session = None

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.cleanup()


def create_github_client(github_token: str) -> MCPClient:
    args = [
        "run",
        "-i",
        "--rm",
        "-e",
        "GITHUB_PERSONAL_ACCESS_TOKEN",
    ]

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    if log_level in ("WARNING", "ERROR", "CRITICAL"):
        args.extend(["-e", "GITHUB_LOG_FILE=/dev/null"])

    args.append("ghcr.io/github/github-mcp-server")

    return MCPClient(
        command="docker",
        args=args,
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": github_token},
    )
