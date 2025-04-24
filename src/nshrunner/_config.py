from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import nshconfig as C
import nshsnap

from ._seed import SeedConfig
from ._util.git import gitignored_dir

log = logging.getLogger(__name__)


class Config(C.Config):
    working_dir: str | Path | Literal["cwd", "tmp", "home-cache"] = "home-cache"
    """
    The `working_dir` parameter is a string that represents the directory where the program will save its execution files and logs.
    This is used when submitting the program to a SLURM/LSF cluster or when using the `local_sessions` method.

    Accepted values are:
    - "cwd": The current working directory.
    - "tmp": The temporary directory.
    - "home-cache" (default): The cache directory in the user's home directory (i.e., `~/.cache/nshrunner`).

    """

    seed: int | SeedConfig | None = SeedConfig(seed=0)
    """Seed configuration for the runner."""

    env: Mapping[str, str] | None = None
    """Environment variables to set for the session."""

    snapshot: bool | nshsnap.SnapshotConfig | None = False
    """Snapshot configuration for the session."""

    save_main_script: bool = True
    """Whether to save the main script or notebook that's being executed."""

    save_git_diff: bool = True
    """Whether to save the git diff if the current directory is in a git repository."""

    def _resolve_seed_config(self):
        if self.seed is None:
            return None

        if isinstance(self.seed, int):
            return SeedConfig(seed=self.seed)

        return self.seed

    def _resolve_working_dir(self):
        match self.working_dir:
            case "cwd":
                return Path.cwd() / "nshrunner"
            case "tmp":
                return Path("/tmp") / "nshrunner"
            case "home-cache":
                return Path.home() / ".cache" / "nshrunner"
            case _:
                return Path(self.working_dir)

    def _resolve_snapshot_config(self, session_dir: Path):
        if (snapshot := self.snapshot) in (False, None):
            return None
        if snapshot is True:
            snapshot = nshsnap.SnapshotConfig()

        snapshot = snapshot.model_copy(deep=True)
        if snapshot.snapshot_dir is None:
            snapshot.snapshot_dir = gitignored_dir(session_dir / "nshsnap", create=True)
        return snapshot
