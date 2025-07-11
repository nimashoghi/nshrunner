from __future__ import annotations

from typing import TYPE_CHECKING

from nshsnap import SnapshotConfig as SnapshotConfig
from nshsnap import load_existing_snapshot as load_existing_snapshot
from nshsnap import snapshot as snapshot

from . import session as session
from ._config import Config as Config
from ._runner import Config as RunnerConfig
from ._runner import Runner as Runner
from ._submit._util import SubmissionScript as SubmissionScript
from .session import Session as Session
from .submission import Submission as Submission
from .wrapper_fns import run_local as run_local
from .wrapper_fns import submit_parallel_screen as submit_parallel_screen
from .wrapper_fns import submit_screen as submit_screen
from .wrapper_fns import submit_slurm as submit_slurm

if TYPE_CHECKING:
    _ = RunnerConfig

try:
    from . import configs as configs
except:
    configs = None


try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    # For Python <3.8
    from importlib_metadata import (  # pyright: ignore[reportMissingImports]
        PackageNotFoundError,
        version,
    )

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"
