import asyncio
import logging
from typing import Awaitable, Callable, TypeVar

from voyageai.error import RateLimitError

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def voyage_retry(
    fn: Callable[[], Awaitable[T]], retires: int = 2, wait: int = 60
) -> T:
    for attempt in range(retires):
        try:
            return await fn()

        except RateLimitError:
            if attempt < retires - 1:
                logger.warning("Rate limited, waiting %ds...", wait)
                await asyncio.sleep(wait)
    raise RateLimitError("Still rate limited after retry")
