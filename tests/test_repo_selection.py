from unittest.mock import AsyncMock, MagicMock

import pytest

from repo_lens.app.repo_selection import (
    ChatRepoState,
    handle_command,
    parse_org_name,
    parse_owner_repo,
    repo_ready_message,
    resolve_org_from_arg,
    resolve_repo_from_arg,
    switch_org,
    switch_repo,
)
from repo_lens.core.repo_context import RepoContext


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("openshift-hyperfleet", "openshift-hyperfleet"),
        ("org/repo", None),
        ("", None),
    ],
)
def test_parse_org_name(value: str, expected: str | None) -> None:
    assert parse_org_name(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("owner/repo", ("owner", "repo")),
        ("a/b/c", None),
    ],
)
def test_parse_owner_repo(value: str, expected: tuple[str, str] | None) -> None:
    assert parse_owner_repo(value) == expected


def test_chat_repo_state_set_repo_syncs_active_org() -> None:
    state = ChatRepoState(
        repo_context=RepoContext(owner="old", repo="repo"),
        active_org="old",
    )
    state.set_repo(RepoContext(owner="new", repo="other"))
    assert state.repo_context.key == "new/other"
    assert state.active_org == "new"


def test_repo_ready_message_switched_omits_hint() -> None:
    message = repo_ready_message("org/repo", switched=True)
    assert "Now chatting about" in message
    assert "/org" not in message


@pytest.mark.parametrize(
    ("arg", "expected_org", "error_substr"),
    [
        ("", None, "/org"),
        ("my-org", "my-org", None),
    ],
)
@pytest.mark.asyncio
async def test_resolve_org_from_arg(
    arg: str, expected_org: str | None, error_substr: str | None
) -> None:
    org, err = await resolve_org_from_arg(arg)
    assert org == expected_org
    if error_substr is not None:
        assert err is not None
        assert error_substr in err
    else:
        assert err is None


def _mock_app(validate: bool = True) -> MagicMock:
    app = MagicMock()
    app.validate_repo = AsyncMock(return_value=validate)
    return app


@pytest.mark.parametrize(
    ("arg", "current_owner", "validate", "expected_ctx", "error_substr"),
    [
        (
            "owner/repo",
            None,
            True,
            RepoContext(owner="owner", repo="repo"),
            None,
        ),
        (
            "my-repo",
            "my-org",
            True,
            RepoContext(owner="my-org", repo="my-repo"),
            None,
        ),
        ("my-repo", None, True, None, "/org"),
        ("owner/repo", None, False, None, "Could not access"),
    ],
)
@pytest.mark.asyncio
async def test_resolve_repo_from_arg(
    arg: str,
    current_owner: str | None,
    validate: bool,
    expected_ctx: RepoContext | None,
    error_substr: str | None,
) -> None:
    app = _mock_app(validate=validate)
    ctx, err = await resolve_repo_from_arg(app, arg, current_owner=current_owner)
    assert ctx == expected_ctx
    if error_substr is not None:
        assert err is not None
        assert error_substr in err
    else:
        assert err is None
    if expected_ctx is not None:
        app.validate_repo.assert_awaited_once_with(expected_ctx)


class MessageRecorder:
    def __init__(self) -> None:
        self.messages: list[tuple[str, bool]] = []

    async def __call__(self, message: str, *, error: bool = False) -> None:
        self.messages.append((message, error))


@pytest.mark.asyncio
async def test_switch_org_bad_arg_reports_error() -> None:
    state = ChatRepoState.from_context(RepoContext(owner="org", repo="repo"))
    recorder = MessageRecorder()
    await switch_org(state, "", on_message=recorder)
    assert state.active_org == "org"
    assert len(recorder.messages) == 1
    assert recorder.messages[0][1] is True
    assert "/org" in recorder.messages[0][0]


@pytest.mark.asyncio
async def test_switch_org_updates_state() -> None:
    state = ChatRepoState.from_context(RepoContext(owner="org", repo="repo"))
    recorder = MessageRecorder()
    await switch_org(state, "new-org", on_message=recorder)
    assert state.active_org == "new-org"
    assert len(recorder.messages) == 1
    assert recorder.messages[0][1] is False
    assert "new-org" in recorder.messages[0][0]


@pytest.mark.asyncio
async def test_switch_repo_updates_state() -> None:
    app = _mock_app()
    state = ChatRepoState.from_context(RepoContext(owner="org", repo="old"))
    recorder = MessageRecorder()
    await switch_repo(app, state, "new-repo", on_message=recorder)
    assert state.repo_context == RepoContext(owner="org", repo="new-repo")
    assert state.active_org == "org"
    assert len(recorder.messages) == 1
    assert recorder.messages[0][1] is False
    assert "org/new-repo" in recorder.messages[0][0]


@pytest.mark.asyncio
async def test_handle_command_unknown_returns_false() -> None:
    app = _mock_app()
    state = ChatRepoState.from_context(RepoContext(owner="org", repo="repo"))
    recorder = MessageRecorder()
    handled = await handle_command(app, state, "/foo", "", recorder)
    assert handled is False
    assert recorder.messages == []


@pytest.mark.parametrize("command", ["/clear-cache", "/org", "/repo"])
@pytest.mark.asyncio
async def test_handle_command_known_returns_true(command: str) -> None:
    app = _mock_app()
    app.clear_cache = AsyncMock(return_value=3)
    state = ChatRepoState.from_context(RepoContext(owner="org", repo="repo"))
    recorder = MessageRecorder()
    arg = "other-org" if command == "/org" else "other-repo"
    handled = await handle_command(app, state, command, arg, recorder)
    assert handled is True
    assert len(recorder.messages) >= 1


@pytest.mark.asyncio
async def test_handle_command_org_updates_active_org() -> None:
    app = _mock_app()
    state = ChatRepoState.from_context(RepoContext(owner="org", repo="repo"))
    recorder = MessageRecorder()
    handled = await handle_command(app, state, "/org", "other-org", recorder)
    assert handled is True
    assert state.active_org == "other-org"
