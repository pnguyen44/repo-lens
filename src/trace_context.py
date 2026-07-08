import structlog
from uuid import uuid4


def start_query_trace() -> str:
    structlog.contextvars.clear_contextvars()
    query_id = uuid4().hex[:8]

    structlog.contextvars.bind_contextvars(query_id=query_id)
    return query_id
