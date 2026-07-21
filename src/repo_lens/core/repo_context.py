from dataclasses import dataclass


@dataclass(frozen=True)
class RepoContext:
    owner: str
    repo: str

    @property
    def key(self) -> str:
        return f"{self.owner}/{self.repo}"
