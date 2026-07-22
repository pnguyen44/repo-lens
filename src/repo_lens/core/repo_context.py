from dataclasses import dataclass


@dataclass(frozen=True)
class RepoContext:
    owner: str
    repo: str

    @property
    def key(self) -> str:
        return f"{self.owner}/{self.repo}"

    def prompt_suffix(self) -> str:
        return (
            f"\n\nActive repository: {self.key}. "
            "Never ask the user for owner or repository name. "
            "Use this repo for tools, links, and answers."
        )
