import chainlit as cl

from repo_lens.agents.orchestrator import delegation_label
from repo_lens.app.runtime import App
from repo_lens.core.config import create_config
from repo_lens.core.mcp_client import create_github_client
from repo_lens.core.repo_context import RepoContext


@cl.on_chat_start  # type: ignore[misc]
async def on_chat_start() -> None:
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
    cl.user_session.set("repo_context", repo_context)

    app.document_indexer.load_from_store()
    await app.ensure_indexed(repo_context=repo_context)

    await cl.Message(
        content=f"Chatting about `{repo_context.key}`. Ask a question about this repo."
    ).send()


@cl.on_message  # type: ignore[misc]
async def on_message(message: cl.Message) -> None:
    app = cl.user_session.get("app")
    repo_context = cl.user_session.get("repo_context")

    async def on_delegate(agent_name: str, task: str) -> None:
        async with cl.Step(name=delegation_label(agent_name), type="tool") as step:
            step.output = task

    before = app.token_tracker.summary()
    answer = await app.orchestrator.run(
        query=message.content, repo_context=repo_context, on_delegate=on_delegate
    )

    await cl.Message(content=answer).send()

    await cl.Message(content=f"_Tokens: {app.format_tokens_for_turn(before)}_").send()


@cl.on_chat_end  # type: ignore[misc]
async def on_chat_end() -> None:
    github_mcp = cl.user_session.get("github_mcp")

    if github_mcp is not None:
        await github_mcp.cleanup()
