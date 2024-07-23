from pathlib import Path


def _run_git_pre_commit_hook(self):
    git_dir = self._find_git_dir()
    if not git_dir:
        log.info("Not a git repository. Skipping pre-commit hook.")
        return True

    pre_commit_hook = git_dir / "hooks" / "pre-commit"
    if not pre_commit_hook.exists():
        log.info("No pre-commit hook found. Skipping.")
        return True

    try:
        result = subprocess.run(
            [str(pre_commit_hook)],
            check=True,
            capture_output=True,
            text=True,
            cwd=git_dir.parent,
        )
        log.info("Git pre-commit hook passed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"Git pre-commit hook failed. Output:\n{e.stdout}\n{e.stderr}")
        return False


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
