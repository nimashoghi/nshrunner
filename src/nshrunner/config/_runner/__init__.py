from __future__ import annotations

__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from nshrunner._runner import Config as Config
    from nshrunner._runner import PythonLoggingConfig as PythonLoggingConfig
    from nshrunner._runner import SeedConfig as SeedConfig
else:

    def __getattr__(name):
        import importlib

        if name in globals():
            return globals()[name]
        if name == "PythonLoggingConfig":
            return importlib.import_module("nshrunner._runner").PythonLoggingConfig
        if name == "RunnerConfig":
            return importlib.import_module("nshrunner._runner").RunnerConfig
        if name == "SeedConfig":
            return importlib.import_module("nshrunner._runner").SeedConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Submodule exports
