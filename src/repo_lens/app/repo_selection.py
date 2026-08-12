from typing import Protocol

from repo_lens.app.runtime import App
from repo_lens.core.config import Config
from repo_lens.core.repo_context import RepoContext

REPO_SWITCH_HINT = "Use `/repo owner/repo` to switch repos."
REPO_PUBLIC_NOTE = "Only public repositories are accessible."


class OnStatusCallback(Protocol):
    async def __call__(self, label: str, message: str) -> None: ...


async def bootstrap_repo(
    app: App, config: Config
) -> tuple[RepoContext, None] | tuple[None, str]:
    owner = config.default_org
    repo = config.default_repo

    if owner is None or repo is None:
        return None, "DEFAULT_ORG and DEFAULT_REPO must be set in .env"

    repo_context = RepoContext(owner=owner, repo=repo)
    if not await app.validate_repo(repo_context):
        return (
            None,
            f"Could not access `{repo_context.key}`. Check your .env and GitHub token.",
        )

    return repo_context, None


async def setup_repo(
    app: App, config: Config, on_status: OnStatusCallback
) -> RepoContext | None:
    repo_context, err = await bootstrap_repo(app=app, config=config)
    if err is not None:
        await on_status(label="Error", message=err)
        return None
    if repo_context is None:
        return None

    await on_status(label="Status", message="Loading repository...")
    app.document_indexer.sync_bm25_from_store()

    await on_status(label="Ready", message=repo_ready_message(repo_context.key))
    return repo_context


def repo_ready_message(repo_key: str, switched: bool = False) -> str:
    label = "Now chatting about" if switched else "Chatting about"
    message = f"{label} `{repo_key}`."
    if not switched:
        message = f"{message} {REPO_SWITCH_HINT} {REPO_PUBLIC_NOTE}"
    return message


def parse_owner_repo(value: str) -> tuple[str, str] | None:
    cleaned = value.strip()
    if cleaned.count("/") != 1:
        return None

    owner, repo = cleaned.split("/", 1)
    owner = owner.strip()
    repo = repo.strip()

    if not owner or not repo:
        return None

    return owner, repo


async def resolve_repo_from_arg(
    app: App, arg: str
) -> tuple[RepoContext, None] | tuple[None, str]:
    parsed = parse_owner_repo(arg)
    if parsed is None:
        return (
            None,
            "Usage: /repo owner/repo (e.g. openshift-hyperfleet/hyperfleet-api)",
        )

    owner, repo = parsed
    repo_context = RepoContext(owner=owner, repo=repo)
    if await app.validate_repo(repo_context):
        return repo_context, None

    return None, (f"Could not access `{owner}/{repo}`. Check the name and try again.")
