import asyncio
import logging

logger = logging.getLogger(__name__)


async def wait_for_retry(retries: int, max_retries: int = 3) -> bool:
    """Sleep with exponential backoff before a rate-limit retry.

    Returns True if retries are exhausted and the caller should give up.
    """
    if retries >= max_retries:
        logger.error("Rate limited after %d retries. Skipping.", retries)
        return True

    wait = 2**retries
    logger.warning("Rate limited. Retrying in %ds...", wait)
    await asyncio.sleep(wait)
    return False
