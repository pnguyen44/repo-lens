from dataclasses import dataclass
from typing import Protocol, Self

from repo_lens.app.runtime import App
from repo_lens.core.config import Config
from repo_lens.core.repo_context import RepoContext

REPO_SWITCH_HINT = "Switch org: `/org <name>`. Switch repo: `/repo <repo-name>`, or `/repo owner/repo`."
REPO_PUBLIC_NOTE = "Only public repositories are accessible."


class OnMessageCallback(Protocol):
    async def __call__(self, message: str, *, error: bool = False) -> None: ...


@dataclass
class ChatRepoState:
    repo_context: RepoContext
    active_org: str

    @classmethod
    def from_context(cls, repo_context: RepoContext) -> Self:
        return cls(repo_context=repo_context, active_org=repo_context.owner)

    def set_org(self, org: str) -> None:
        self.active_org = org

    def set_repo(self, repo_context: RepoContext) -> None:
        self.repo_context = repo_context
        self.active_org = repo_context.owner


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
    app: App, config: Config, on_message: OnMessageCallback
) -> RepoContext | None:
    repo_context, err = await bootstrap_repo(app=app, config=config)
    if err is not None:
        await on_message(err, error=True)
        return None
    if repo_context is None:
        return None

    await on_message("Loading repository...")
    await app.document_indexer.sync_bm25_from_store()

    await on_message(repo_ready_message(repo_context.key))
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


def parse_org_name(value: str) -> str | None:
    cleaned = value.strip()
    if not cleaned or "/" in cleaned:
        return None

    return cleaned


async def resolve_org_from_arg(arg: str) -> tuple[str, None] | tuple[None, str]:
    org = parse_org_name(arg)
    if org is None:
        return None, "Usage: /org <name> (e.g. openshift-hyperfleet)"

    return org, None


async def resolve_repo_from_arg(
    app: App, arg: str, current_owner: str | None = None
) -> tuple[RepoContext, None] | tuple[None, str]:
    cleaned = arg.strip()
    if not cleaned:
        return (
            None,
            "Usage: /repo owner/repo or /repo <repo-name> after /org <name>",
        )

    if "/" in cleaned:
        parsed = parse_owner_repo(cleaned)
        if parsed is None:
            return (
                None,
                "Usage: /repo owner/repo (e.g. openshift-hyperfleet/hyperfleet-api)",
            )

        owner, repo = parsed
    elif current_owner:
        owner, repo = current_owner, cleaned
    else:
        return (
            None,
            "Usage: /repo owner/repo or set org first with /org <name>",
        )

    repo_context = RepoContext(owner=owner, repo=repo)
    if await app.validate_repo(repo_context):
        return repo_context, None

    return None, f"Could not access `{owner}/{repo}`. Check the name and try again."


async def switch_org(
    state: ChatRepoState, arg: str, on_message: OnMessageCallback
) -> None:
    org, err = await resolve_org_from_arg(arg)
    if err is not None:
        await on_message(err, error=True)
        return
    if org is not None:
        state.set_org(org)
        await on_message(
            f"Active org set to `{org}`. Use `/repo <repo-name>` to switch."
        )


async def switch_repo(
    app: App, state: ChatRepoState, arg: str, on_message: OnMessageCallback
) -> None:
    new_repo_context, err = await resolve_repo_from_arg(
        app=app, arg=arg, current_owner=state.active_org
    )
    if err is not None:
        await on_message(err, error=True)
        return
    if new_repo_context is not None:
        state.set_repo(new_repo_context)
        app.orchestrator.reset_conversation(
            note=f"[System: repository switched to {new_repo_context.key}]"
        )
        await on_message(repo_ready_message(state.repo_context.key, switched=True))


async def clear_cache(
    app: App, state: ChatRepoState, on_message: OnMessageCallback
) -> None:
    key = state.repo_context.key
    removed = await app.clear_cache(repo_context=state.repo_context)
    await on_message(f"Cleared `{key}` ({removed} chunks removed).")


async def handle_command(
    app: App,
    state: ChatRepoState,
    command: str,
    arg: str,
    on_message: OnMessageCallback,
) -> bool:
    match command:
        case "/clear-cache":
            await clear_cache(app=app, state=state, on_message=on_message)
            return True
        case "/org":
            await switch_org(state=state, arg=arg, on_message=on_message)
            return True
        case "/repo":
            await switch_repo(app=app, state=state, arg=arg, on_message=on_message)
            return True
    return False
