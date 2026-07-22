import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()


@dataclass
class Config:
    provider: str
    model: str
    github_token: str
    default_org: str | None
    default_repo: str | None
    voyage_embed_model: str
    voyage_rerank_model: str
    chroma_host: str
    chroma_port: int


def create_config() -> Config:
    provider = os.environ.get("PROVIDER", "anthropic")
    model = os.environ.get("MODEL", "")
    github_token = os.environ.get("GITHUB_TOKEN", "")
    default_org = os.environ.get("DEFAULT_ORG") or None
    default_repo = os.environ.get("DEFAULT_REPO") or None
    voyage_embed_model = os.environ.get("VOYAGE_EMBED_MODEL", "voyage-3-large")
    voyage_rerank_model = os.environ.get("VOYAGE_RERANK_MODEL", "rerank-2")
    chroma_host = os.environ.get("CHROMA_HOST", "localhost")
    chroma_port = int(os.environ.get("CHROMA_PORT", "8000"))

    if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY", ""):
        raise ValueError("ANTHROPIC_API_KEY is missing a value. Update .env")
    if provider == "gemini" and not os.environ.get("GEMINI_API_KEY", ""):
        raise ValueError("GEMINI_API_KEY is missing a value. Update .env")
    if not model:
        raise ValueError("MODEL is missing a value. Update .env")
    if not github_token:
        raise ValueError("GITHUB_TOKEN is missing a value. Update .env")

    return Config(
        provider=provider,
        model=model,
        github_token=github_token,
        default_org=default_org,
        default_repo=default_repo,
        voyage_embed_model=voyage_embed_model,
        voyage_rerank_model=voyage_rerank_model,
        chroma_host=chroma_host,
        chroma_port=chroma_port,
    )
