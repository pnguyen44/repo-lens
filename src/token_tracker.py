from typing import TypedDict, NotRequired


class UsagePayload(TypedDict):
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: NotRequired[int]
    cache_creation_input_tokens: NotRequired[int]


class TokenTracker:
    def __init__(self) -> None:
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.cache_read_input_tokens: int | None = None
        self.cache_creation_input_tokens: int | None = None
        self.request_count: int = 0

    def record(self, usage: UsagePayload) -> None:
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)
        if "cache_read_input_tokens" in usage:
            self.cache_read_input_tokens = (self.cache_read_input_tokens or 0) + usage[
                "cache_read_input_tokens"
            ]
        if "cache_creation_input_tokens" in usage:
            self.cache_creation_input_tokens = (
                self.cache_creation_input_tokens or 0
            ) + usage["cache_creation_input_tokens"]
        self.request_count += 1

    def summary(self) -> dict[str, int]:
        data = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "request_count": self.request_count,
        }

        return {k: v for k, v in data.items() if v is not None}
