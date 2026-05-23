import os
from dotenv import load_dotenv

load_dotenv()


def create_config() -> dict[str, str]:
    config: dict[str, str] = {}

    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    claude_model = os.environ.get("CLAUDE_MODEL", "")

    if not claude_model:
        raise ValueError("CLAUDE_MODEL is missing a value. Update .env")
    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY is missing a value. Update .env")

    config["claude_model"] = claude_model

    return config
