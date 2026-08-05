import asyncio
import logging
import os

import chainlit as cl
from google.genai.errors import ClientError as GeminiClientError

from repo_lens.agents.orchestrator import delegation_label
from repo_lens.app.runtime import App
from repo_lens.core.config import create_config
from repo_lens.core.mcp_client import create_github_client
from repo_lens.core.repo_context import RepoContext
from repo_lens.core.retry import format_rate_limit_message
from repo_lens.providers.chat_client import StreamError

logger = logging.getLogger(__name__)


def _validate_auth_env() -> None:
    if not os.environ.get("APP_USER") or not os.environ.get("APP_PASS"):
        raise RuntimeError(
            "APP_USER and APP_PASS must be set and non-empty in the environment"
        )


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

    config = create_config()
    github_mcp = create_github_client(config.github_token)
    await github_mcp.connect()
    cl.user_session.set("github_mcp", github_mcp)

    owner = config.default_org
    repo = config.default_repo
    if owner is None or repo is None:
        raise ValueError("DEFAULT_ORG and DEFAULT_REPO must be set in .env")

    app = App(config=config, github_mcp=github_mcp)
    cl.user_session.set("app", app)

    repo_context = RepoContext(owner=owner, repo=repo)

    if not await app.validate_repo(repo_context):
        msg.content = (
            f"Could not access `{repo_context.key}`. Check that DEFAULT_ORG/DEFAULT_REPO "
            "are correct and the GitHub token has access."
        )
        await msg.update()
        cl.user_session.set("setup_error", msg.content)
        return

    cl.user_session.set("repo_context", repo_context)

    msg.content = "Loading repository..."
    await msg.update()

    app.document_indexer.load_from_store()

    msg.content = (
        f"Chatting about `{repo_context.key}`. Ask a question about this repo."
    )
    await msg.update()
    startup_ready.set()


async def ui_on_delegate(agent_name: str, task: str) -> None:
    async with cl.Step(name=delegation_label(agent_name), type="tool") as step:
        step.output = task


@cl.on_message  # type: ignore[misc]
async def on_message(message: cl.Message) -> None:
    setup_error = cl.user_session.get("setup_error")
    if setup_error:
        await cl.Message(content=setup_error).send()
        return

    startup_ready: asyncio.Event | None = cl.user_session.get("startup_ready")
    if startup_ready is None or not startup_ready.is_set():
        await cl.Message(content="Still setting up. Please wait a moment.").send()
        return

    lock: asyncio.Lock = cl.user_session.get("message_lock")

    if lock.locked():
        await cl.Message(
            content="Still working on your previous message. Please wait for it to finish"
        ).send()

        return

    async with lock:
        app = cl.user_session.get("app")
        repo_context = cl.user_session.get("repo_context")

        before = app.token_tracker.summary()

        try:

            async def on_file_fetched(path: str) -> None:
                await app.index_file_if_needed(repo_context, path)

            answer = await app.orchestrator.run(
                query=message.content,
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
        await cl.Message(
            content=f"_Tokens: {app.format_tokens_for_turn(before)}_"
        ).send()


@cl.on_chat_end  # type: ignore[misc]
async def on_chat_end() -> None:
    github_mcp = cl.user_session.get("github_mcp")

    if github_mcp is not None:
        await github_mcp.cleanup()
