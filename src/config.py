import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()


@dataclass
class Config:
    claude_model: str
    github_token: str
    default_org: str | None
    voyage_embed_model: str
    voyage_rerank_model: str


def create_config() -> Config:
    claude_model = os.environ.get("CLAUDE_MODEL", "")
    github_token = os.environ.get("GITHUB_TOKEN", "")
    default_org = os.environ.get("DEFAULT_ORG") or None
    voyage_embed_model = os.environ.get("VOYAGE_EMBED_MODEL", "voyage-3-large")
    voyage_rerank_model = os.environ.get("VOYAGE_RERANK_MODEL", "rerank-2")

    if not os.environ.get("ANTHROPIC_API_KEY", ""):
        raise ValueError("ANTHROPIC_API_KEY is missing a value. Update .env")
    if not claude_model:
        raise ValueError("CLAUDE_MODEL is missing a value. Update .env")
    if not github_token:
        raise ValueError("GITHUB_TOKEN is missing a value. Update .env")

    return Config(
        claude_model=claude_model,
        github_token=github_token,
        default_org=default_org,
        voyage_embed_model=voyage_embed_model,
        voyage_rerank_model=voyage_rerank_model,
    )
