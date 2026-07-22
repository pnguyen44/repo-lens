from repo_lens.providers.token_tracker import TokenCounts, TokenTracker


def test_record_once_no_cache() -> None:
    tracker = TokenTracker()
    tracker.record({"input_tokens": 100, "output_tokens": 50})

    assert tracker.summary() == {
        "input_tokens": 100,
        "output_tokens": 50,
        "request_count": 1,
    }


def test_record_accumulates() -> None:
    tracker = TokenTracker()
    tracker.record({"input_tokens": 100, "output_tokens": 50})
    tracker.record(
        {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 10,
            "cache_creation_input_tokens": 10,
        }
    )

    assert tracker.summary() == {
        "input_tokens": 200,
        "output_tokens": 100,
        "cache_read_input_tokens": 10,
        "cache_creation_input_tokens": 10,
        "request_count": 2,
    }


def test_record_with_all_cache_fields() -> None:
    tracker = TokenTracker()
    tracker.record(
        {
            "input_tokens": 50,
            "output_tokens": 25,
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 20,
        }
    )

    assert tracker.summary() == {
        "input_tokens": 50,
        "output_tokens": 25,
        "cache_read_input_tokens": 30,
        "cache_creation_input_tokens": 20,
        "request_count": 1,
    }


def test_token_delta_basic() -> None:
    before: TokenCounts = {
        "input_tokens": 100,
        "output_tokens": 50,
        "request_count": 1,
    }
    after: TokenCounts = {
        "input_tokens": 250,
        "output_tokens": 120,
        "request_count": 3,
    }
    assert TokenTracker.token_delta(before, after) == {
        "input_tokens": 150,
        "output_tokens": 70,
        "request_count": 2,
    }


def test_token_delta_with_cache() -> None:
    before: TokenCounts = {
        "input_tokens": 100,
        "output_tokens": 50,
        "request_count": 1,
    }
    after: TokenCounts = {
        "input_tokens": 200,
        "output_tokens": 80,
        "request_count": 2,
        "cache_read_input_tokens": 40,
        "cache_creation_input_tokens": 10,
    }
    assert TokenTracker.token_delta(before, after) == {
        "input_tokens": 100,
        "output_tokens": 30,
        "request_count": 1,
        "cache_read_input_tokens": 40,
        "cache_creation_input_tokens": 10,
    }


def test_token_delta_empty_before() -> None:
    before: TokenCounts = {
        "input_tokens": 0,
        "output_tokens": 0,
        "request_count": 0,
    }
    after: TokenCounts = {
        "input_tokens": 100,
        "output_tokens": 50,
        "request_count": 1,
        "cache_read_input_tokens": 20,
    }
    assert TokenTracker.token_delta(before, after) == after


def test_token_delta_cache_in_before_and_after() -> None:
    before: TokenCounts = {
        "input_tokens": 100,
        "output_tokens": 50,
        "request_count": 1,
        "cache_read_input_tokens": 10,
        "cache_creation_input_tokens": 5,
    }
    after: TokenCounts = {
        "input_tokens": 250,
        "output_tokens": 90,
        "request_count": 3,
        "cache_read_input_tokens": 40,
        "cache_creation_input_tokens": 15,
    }
    assert TokenTracker.token_delta(before, after) == {
        "input_tokens": 150,
        "output_tokens": 40,
        "request_count": 2,
        "cache_read_input_tokens": 30,
        "cache_creation_input_tokens": 10,
    }


def test_summary_empty() -> None:
    tracker = TokenTracker()
    assert tracker.summary() == {
        "input_tokens": 0,
        "output_tokens": 0,
        "request_count": 0,
    }
