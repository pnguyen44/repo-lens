import asyncio

import pytest

from repo_lens.core.retry import (
    format_rate_limit_message,
    parse_retry_after_seconds,
    wait_for_retry,
)


@pytest.mark.parametrize(
    ("detail", "expected"),
    [
        ("Please retry in 30.5s", 31),
        ("RETRY IN 1s", 1),
        ("Quota exceeded. retry in 0.5s", 1),
        ("Something went wrong", None),
        ("", None),
    ],
)
def test_parse_retry_after_seconds(detail: str, expected: int | None) -> None:
    assert parse_retry_after_seconds(detail) == expected


@pytest.mark.parametrize(
    ("detail", "expected"),
    [
        (
            "429 RESOURCE_EXHAUSTED. Quota exceeded. Please retry in 30s.",
            "The AI service is temporarily unavailable due to a usage limit. "
            "Please try again in about 30 seconds.",
        ),
        (
            "Rate limit exceeded",
            "The AI service is temporarily unavailable due to a usage limit. "
            "Please wait a minute and try again.",
        ),
        (
            "Connection reset by peer",
            "The AI service failed to respond. Please wait a moment and try again.",
        ),
    ],
)
def test_format_rate_limit_message(detail: str, expected: str) -> None:
    assert format_rate_limit_message(detail) == expected


@pytest.mark.asyncio
async def test_wait_for_retry_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    assert await wait_for_retry(retries=3, max_retries=3) is True
    assert slept == []


@pytest.mark.asyncio
async def test_wait_for_retry_uses_detail_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    assert (
        await wait_for_retry(
            retries=0,
            detail="Quota exceeded. Please retry in 5s.",
        )
        is False
    )
    assert slept == [5.0]


@pytest.mark.asyncio
async def test_wait_for_retry_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    assert await wait_for_retry(retries=2) is False
    assert slept == [4.0]
