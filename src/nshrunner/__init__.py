from __future__ import annotations

from typing import TYPE_CHECKING

from nshsnap import SnapshotConfig as SnapshotConfig
from nshsnap import load_existing_snapshot as load_existing_snapshot
from nshsnap import snapshot as snapshot

from . import session as session
from ._runner import Config as Config
from ._runner import Config as RunnerConfig
from ._runner import Runner as Runner
from ._submit._util import Submission as Submission
from .session import Session as Session
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
