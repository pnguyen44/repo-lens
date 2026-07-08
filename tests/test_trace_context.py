import structlog
from trace_context import start_query_trace


def test_start_query_trace_return_8_char_hex():
    query_id = start_query_trace()

    assert len(query_id) == 8
    assert all(c in "0123456789abcdef" for c in query_id)


def test_start_query_trace_return_unique_id():
    query_id_1 = start_query_trace()
    query_id_2 = start_query_trace()

    assert query_id_1 != query_id_2


def test_query_id_bound_in_contextvars():
    query_id = start_query_trace()

    ctx = structlog.contextvars.get_contextvars()
    assert ctx["query_id"] == query_id
