import asyncio
import logging
import os

import chainlit as cl
from google.genai.errors import ClientError as GeminiClientError

from repo_lens.agents.orchestrator import delegation_label
from repo_lens.app.repo_selection import (
    repo_ready_message,
    resolve_repo_from_arg,
    setup_repo,
)
from repo_lens.app.runtime import App
from repo_lens.core.config import create_config
from repo_lens.core.logging_config import configure_logging
from repo_lens.core.mcp_client import create_github_client
from repo_lens.core.repo_context import RepoContext
from repo_lens.core.retry import format_rate_limit_message
from repo_lens.providers.chat_client import StreamError

logger = logging.getLogger(__name__)


async def ui_send_repo_status(repo_key: str, *, switched: bool = False) -> None:
    await cl.Message(content=repo_ready_message(repo_key, switched=switched)).send()


def _validate_auth_env() -> None:
    if not os.environ.get("APP_USER") or not os.environ.get("APP_PASS"):
        raise RuntimeError(
            "APP_USER and APP_PASS must be set and non-empty in the environment"
        )


configure_logging(os.getenv("LOG_LEVEL", "INFO"))
_validate_auth_env()


@cl.password_auth_callback  # type: ignore[misc]
def auth_callback(username: str, password: str) -> cl.User | None:
    expected_user = os.environ.get("APP_USER", "")
    expected_pass = os.environ.get("APP_PASS", "")
    if not username or not password:
        return None
    if not expected_user or not expected_pass:
        return None
    if username == expected_user and password == expected_pass:
        return cl.User(identifier=username)
    return None


@cl.on_chat_start  # type: ignore[misc]
async def on_chat_start() -> None:
    cl.user_session.set("message_lock", asyncio.Lock())
    startup_ready = asyncio.Event()
    cl.user_session.set("startup_ready", startup_ready)

    msg = cl.Message(content="Setting up...")
    await msg.send()

    try:
        config = create_config()
        github_mcp = create_github_client(config.github_token)
        await github_mcp.connect()
        cl.user_session.set("github_mcp", github_mcp)

        app = App(config=config, github_mcp=github_mcp)
        cl.user_session.set("app", app)

        async def on_status(label: str, message: str) -> None:
            msg.content = message
            await msg.update()
            if label == "Error":
                cl.user_session.set("setup_error", message)

        repo_context = await setup_repo(app=app, config=config, on_status=on_status)
        if repo_context is None:
            return

        cl.user_session.set("repo_context", repo_context)
        startup_ready.set()
    except Exception as e:
        logger.exception("Startup failed: %s", e)
        error_msg = "Setup failed. Please refresh and try again."
        msg.content = error_msg
        await msg.update()
        cl.user_session.set("setup_error", error_msg)


async def ui_on_delegate(agent_name: str, task: str) -> None:
    async with cl.Step(name=delegation_label(agent_name), type="tool") as step:
        step.output = task


async def ui_clear_cache(app: App, repo_context: RepoContext) -> None:
    removed = await app.clear_cache(repo_context)
    await cl.Message(
        content=f"Cleared `{repo_context.key}` ({removed} chunks removed).",
    ).send()


async def _reject_if_not_ready() -> bool:
    setup_error = cl.user_session.get("setup_error")
    if setup_error:
        await cl.Message(content=setup_error).send()
        return True

    startup_ready: asyncio.Event | None = cl.user_session.get("startup_ready")
    if startup_ready is None or not startup_ready.is_set():
        await cl.Message(content="Still setting up. Please wait a moment.").send()
        return True

    lock: asyncio.Lock = cl.user_session.get("message_lock")

    if lock.locked():
        await cl.Message(
            content="Still working on your previous message. Please wait for it to finish"
        ).send()
        return True

    return False


async def ui_run_query(app: App, repo_context: RepoContext, query: str) -> None:
    before = app.token_tracker.summary()

    try:

        async def on_file_fetched(path: str) -> None:
            await app.index_file_if_needed(repo_context, path)

        answer = await app.orchestrator.run(
            query=query,
            repo_context=repo_context,
            on_delegate=ui_on_delegate,
            on_file_fetched=on_file_fetched,
        )
    except StreamError as e:
        logger.error("Provider stream error: %s", e)
        await cl.Message(content=format_rate_limit_message(str(e))).send()
        return
    except GeminiClientError as e:
        if e.code != 429:
            raise
        detail = getattr(e, "message", str(e))
        logger.error("Gemini rate limit: %s", detail)
        await cl.Message(content=format_rate_limit_message(detail)).send()
        return
    except Exception as e:
        logger.exception("Unexpected chat error: %s", e)
        await cl.Message(
            content="Something went wrong. Please try again in a moment."
        ).send()
        return

    await cl.Message(content=answer or "No response generated.").send()
    await cl.Message(content=f"_Tokens: {app.format_tokens_for_turn(before)}_").send()


async def ui_switch_repo(app: App, arg: str) -> None:
    repo_context, err = await resolve_repo_from_arg(app=app, arg=arg)
    if err is not None:
        await cl.Message(content=err).send()
        return
    if repo_context is None:
        return

    cl.user_session.set("repo_context", repo_context)
    await ui_send_repo_status(repo_context.key, switched=True)


@cl.on_message  # type: ignore[misc]
async def on_message(message: cl.Message) -> None:
    if await _reject_if_not_ready():
        return

    lock: asyncio.Lock = cl.user_session.get("message_lock")

    async with lock:
        app = cl.user_session.get("app")
        repo_context = cl.user_session.get("repo_context")

        stripped = message.content.strip()
        if not stripped:
            return

        parts = stripped.split(maxsplit=1)
        command = parts[0].lower()

        match command:
            case "/clear-cache":
                await ui_clear_cache(app, repo_context)
            case "/repo":
                arg = parts[1] if len(parts) > 1 else ""
                await ui_switch_repo(app=app, arg=arg)
            case _:
                await ui_run_query(
                    app=app, repo_context=repo_context, query=message.content
                )


@cl.on_chat_end  # type: ignore[misc]
async def on_chat_end() -> None:
    github_mcp = cl.user_session.get("github_mcp")

    if github_mcp is not None:
        await github_mcp.cleanup()
