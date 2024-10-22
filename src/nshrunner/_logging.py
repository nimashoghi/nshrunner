from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeAlias

import nshconfig as C
import nshutils

LogLevel: TypeAlias = Literal[
    "CRITICAL", "FATAL", "ERROR", "WARN", "WARNING", "INFO", "DEBUG"
]


class PythonLoggingConfig(C.Config):
    log_level: LogLevel = "INFO"
    """Log level to use for the Python logger."""
    log_save_dir: Path | None = None
    """Directory to save logs to. If None, logs will not be saved."""

    treescope: bool = True
    """If enabled, will use the treescope library to visualize data structures in notebooks."""
    treescope_autovisualize_arrays: bool = True
    """If enabled, will automatically visualize arrays with treescope (if `treescope` is enabled)."""

    lovely_tensors: bool = False
    """If enabled, will use the lovely-tensors library to format PyTorch tensors. False by default as it causes issues when used with `torch.vmap`."""
    lovely_numpy: bool = False
    """If enabled, will use the lovely-numpy library to format numpy arrays. False by default as it causes some issues with other libaries."""

    rich: bool = False
    """If enabled, will use the rich library to format the Python logger output."""
    rich_tracebacks: bool = True
    """If enabled, will use the rich library to format the Python logger tracebacks."""

    def pretty_(
        self,
        *,
        log_level: LogLevel = "INFO",
        torch: bool = True,
        numpy: bool = True,
        rich: bool = True,
        rich_tracebacks: bool = True,
    ):
        self.log_level = log_level
        self.lovely_tensors = torch
        self.lovely_numpy = numpy
        self.rich = rich
        self.rich_tracebacks = rich_tracebacks


def init_python_logging(config: PythonLoggingConfig):
    nshutils.init_python_logging(
        treescope=config.treescope,
        treescope_autovisualize_arrays=config.treescope_autovisualize_arrays,
        lovely_tensors=config.lovely_tensors,
        lovely_numpy=config.lovely_numpy,
        rich=config.rich,
        rich_tracebacks=config.rich_tracebacks,
        log_level=config.log_level,
        log_save_dir=config.log_save_dir,
    )
