from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SubmitOutput:
    command_parts: list[str]
    script_path: Path

    @property
    def command(self) -> str:
        return " ".join(self.command_parts)
