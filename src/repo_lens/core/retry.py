import asyncio
import logging
import math
import re
import time

logger = logging.getLogger(__name__)


def parse_retry_after_seconds(detail: str) -> int | None:
    match = re.search(r"retry in\s+(\d+(?:\.\d+)?)\s*s", detail, re.IGNORECASE)
    if not match:
        return None
    return max(1, math.ceil(float(match.group(1))))


def format_rate_limit_message(detail: str) -> str:
    lower = detail.lower()
    if "quota" not in lower and "rate" not in lower:
        return "The AI service failed to respond. Please wait a moment and try again."

    seconds = parse_retry_after_seconds(detail)
    if seconds is not None:
        return (
            "The AI service is temporarily unavailable due to a usage limit. "
            f"Please try again in about {seconds} seconds."
        )
    return (
        "The AI service is temporarily unavailable due to a usage limit. "
        "Please wait a minute and try again."
    )


def _compute_backoff_seconds(retries: int, detail: str | None) -> int:
    wait = parse_retry_after_seconds(detail) if detail else None
    if wait is None:
        wait = 2**retries
    return wait


def _prepare_retry(retries: int, max_retries: int, detail: str | None) -> int | None:
    if retries >= max_retries:
        logger.error("Rate limited after %d retries. Skipping.", retries)
        return None

    wait = _compute_backoff_seconds(retries=retries, detail=detail)
    logger.warning("Rate limited. Retrying in %ds...", wait)
    return wait


async def wait_for_retry(
    *,
    retries: int,
    max_retries: int = 3,
    detail: str | None = None,
) -> bool:
    """Sleep with exponential backoff before a rate-limit retry.

    Returns True if retries are exhausted and the caller should give up.
    """
    wait = _prepare_retry(retries=retries, max_retries=max_retries, detail=detail)
    if wait is None:
        return True
    await asyncio.sleep(wait)
    return False


def wait_for_retry_sync(
    *,
    retries: int,
    max_retries: int = 3,
    detail: str | None = None,
) -> bool:
    """Sync counterpart to wait_for_retry, for non-async callers like PromptEvaluator."""
    wait = _prepare_retry(retries=retries, max_retries=max_retries, detail=detail)
    if wait is None:
        return True
    time.sleep(wait)
    return False
