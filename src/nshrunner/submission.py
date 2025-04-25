from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import nshsnap

from ._submit._util import SubmissionScript


@dataclass
class Submission:
    id: str
    """The ID of the session."""

    dir_path: Path
    """The path to the session directory."""

    env: dict[str, str] = field(default_factory=lambda: {})
    """Environment variables to set for the session."""

    snapshot: nshsnap.SnapshotInfo | None = None
    """The snapshot information for the session."""

    script: SubmissionScript | None = None
    """The script to run for the session."""
