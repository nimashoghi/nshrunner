from __future__ import annotations

__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from nshrunner._seed import SeedConfig as SeedConfig
else:

    def __getattr__(name):
        import importlib

        if name in globals():
            return globals()[name]
        if name == "SeedConfig":
            return importlib.import_module("nshrunner._seed").SeedConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Submodule exports
