import os
from dotenv import load_dotenv

load_dotenv()


def create_config():
    config = {}

    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not anthropic_api_key:
        raise ValueError("Missing ANTHROPIC_API_KEY")

    config["anthropic-key"] = anthropic_api_key

    return config
