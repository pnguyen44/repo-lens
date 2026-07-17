from typing import Callable, TypeVar
from voyageai.error import RateLimitError
import time


import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


def voyage_retry(fn: Callable[[], T], retires: int = 2, wait: int = 60) -> T:
    for attempt in range(retires):
        try:
            return fn()

        except RateLimitError:
            if attempt < retires - 1:
                logger.warning("Rate limited, waiting %ds...", wait)
                time.sleep(wait)
    raise RateLimitError("Still rate limited after retry")
