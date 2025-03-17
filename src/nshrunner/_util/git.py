from __future__ import annotations

from pathlib import Path


def gitignored_dir(path: Path, *, create: bool = True) -> Path:
    if create:
        path.mkdir(exist_ok=True, parents=True)
    assert path.is_dir(), f"{path} is not a directory"

    gitignore_path = path / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.touch()
        gitignore_path.write_text("*\n")

    return path
