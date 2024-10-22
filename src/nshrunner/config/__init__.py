from __future__ import annotations

__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from nshrunner import PythonLoggingConfig as PythonLoggingConfig
    from nshrunner import RunnerConfig as RunnerConfig
    from nshrunner import SeedConfig as SeedConfig
    from nshrunner import SnapshotConfig as SnapshotConfig
else:

    def __getattr__(name):
        import importlib

        if name in globals():
            return globals()[name]
        if name == "PythonLoggingConfig":
            return importlib.import_module("nshrunner").PythonLoggingConfig
        if name == "RunnerConfig":
            return importlib.import_module("nshrunner").RunnerConfig
        if name == "SeedConfig":
            return importlib.import_module("nshrunner").SeedConfig
        if name == "SnapshotConfig":
            return importlib.import_module("nshrunner").SnapshotConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Submodule exports
from . import _logging as _logging
from . import _runner as _runner
from . import _seed as _seed
