from typing import Protocol

from repo_lens.app.runtime import App
from repo_lens.core.config import Config
from repo_lens.core.repo_context import RepoContext


class AskCallback(Protocol):
    async def __call__(self, prompt: str) -> str | None:
        """Return the user's answer, or None to signal give up (e.g. timeout)."""
        ...


class OnStatusCallback(Protocol):
    async def __call__(self, label: str, message: str) -> None: ...


async def resolve_owner(ask: AskCallback, config: Config) -> str:
    if config.default_org:
        entered = (
            await ask(f"Org [{config.default_org}] (Enter to keep)") or ""
        ).strip()
        return entered or config.default_org
    while True:
        owner = (await ask("Org") or "").strip()
        if owner:
            return owner


async def resolve_repo(
    ask: AskCallback, app: App, config: Config, owner: str
) -> str | None:
    while True:
        if config.default_repo:
            entered = (
                await ask(
                    f"Repo name [{config.default_repo}] (Enter to keep, or /back)"
                )
                or ""
            ).strip()
            if entered.lower() == "/back":
                return None
            repo = entered or config.default_repo
        else:
            repo = (await ask("Repo name (or /back)") or "").strip()
            if repo.lower() == "/back":
                return None

        repo_context = RepoContext(owner=owner, repo=repo)
        if await app.validate_repo(repo_context):
            return repo


async def select_repo(
    app: App, config: Config, ask: AskCallback, on_status: OnStatusCallback
) -> tuple[str, str]:
    while True:
        owner = await resolve_owner(ask=ask, config=config)
        await on_status(label="Org", message=owner)
        repo = await resolve_repo(ask=ask, app=app, config=config, owner=owner)
        if repo is None:
            continue
        return owner, repo
