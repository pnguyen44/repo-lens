from typing import NotRequired, TypedDict


class UsagePayload(TypedDict):
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: NotRequired[int]
    cache_creation_input_tokens: NotRequired[int]


class TokenCounts(TypedDict):
    input_tokens: int
    output_tokens: int
    request_count: int
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

    @staticmethod
    def token_delta(before: TokenCounts, after: TokenCounts) -> TokenCounts:
        delta: TokenCounts = {
            "input_tokens": after["input_tokens"] - before["input_tokens"],
            "output_tokens": after["output_tokens"] - before["output_tokens"],
            "request_count": after["request_count"] - before["request_count"],
        }
        if "cache_read_input_tokens" in after:
            delta["cache_read_input_tokens"] = after[
                "cache_read_input_tokens"
            ] - before.get("cache_read_input_tokens", 0)
        if "cache_creation_input_tokens" in after:
            delta["cache_creation_input_tokens"] = after[
                "cache_creation_input_tokens"
            ] - before.get("cache_creation_input_tokens", 0)
        return delta

    def summary(self) -> TokenCounts:
        result: TokenCounts = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "request_count": self.request_count,
        }
        if self.cache_read_input_tokens is not None:
            result["cache_read_input_tokens"] = self.cache_read_input_tokens
        if self.cache_creation_input_tokens is not None:
            result["cache_creation_input_tokens"] = self.cache_creation_input_tokens
        return result
