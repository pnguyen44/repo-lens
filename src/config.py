import os
from dotenv import load_dotenv

load_dotenv()


def create_config() -> dict[str, str]:
    config: dict[str, str] = {}

    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    claude_model = os.environ.get("CLAUDE_MODEL", "")

    assert claude_model, "Error: CLAUDE_MODEL is missing a value. Update .env"
    assert anthropic_api_key, "Error: ANTHROPIC_API_KEY is missing a value. Update .env"

    config["anthropic-key"] = anthropic_api_key
    config["claude_model"] = claude_model

    return config
