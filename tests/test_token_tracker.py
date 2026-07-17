from repo_lens.providers.token_tracker import TokenTracker


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
