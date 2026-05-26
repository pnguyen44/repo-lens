import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()


@dataclass
class Config:
    claude_model: str
    github_token: str


def create_config() -> Config:
    claude_model = os.environ.get("CLAUDE_MODEL", "")
    github_token = os.environ.get("GITHUB_TOKEN", "")

    if not os.environ.get("ANTHROPIC_API_KEY", ""):
        raise ValueError("ANTHROPIC_API_KEY is missing a value. Update .env")
    if not claude_model:
        raise ValueError("CLAUDE_MODEL is missing a value. Update .env")
    if not github_token:
        raise ValueError("GITHUB_TOKEN is missing a value. Update .env")

    return Config(claude_model=claude_model, github_token=github_token)
