from pathlib import Path


def _find_git_dir(self):
    current_dir = Path.cwd()
    while current_dir != current_dir.parent:
        git_dir = current_dir / ".git"
        if git_dir.is_dir():
            return git_dir
        current_dir = current_dir.parent
    return None


def _gitignored_dir(path: Path, *, create: bool = True) -> Path:
    if create:
        path.mkdir(exist_ok=True, parents=True)
    assert path.is_dir(), f"{path} is not a directory"

    gitignore_path = path / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.touch()
        gitignore_path.write_text("*\n")

    return path
