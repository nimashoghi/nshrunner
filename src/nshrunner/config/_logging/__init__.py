from __future__ import annotations

__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from nshrunner._logging import PythonLoggingConfig as PythonLoggingConfig
else:

    def __getattr__(name):
        import importlib

        if name in globals():
            return globals()[name]
        if name == "PythonLoggingConfig":
            return importlib.import_module("nshrunner._logging").PythonLoggingConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Submodule exports
