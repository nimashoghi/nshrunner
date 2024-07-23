import logging
from pathlib import Path
from typing import Literal, TypeAlias

import nshconfig as C

LogLevel: TypeAlias = Literal[
    "CRITICAL", "FATAL", "ERROR", "WARN", "WARNING", "INFO", "DEBUG"
]


class PythonLoggingConfig(C.Config):
    log_level: LogLevel = "INFO"
    """Log level to use for the Python logger."""
    log_save_dir: Path | None = None
    """Directory to save logs to. If None, logs will not be saved."""

    rich: bool = False
    """If enabled, will use the rich library to format the Python logger output."""
    rich_tracebacks: bool = True
    """If enabled, will use the rich library to format the Python logger tracebacks."""

    lovely_tensors: bool = False
    """If enabled, will use the lovely-tensors library to format PyTorch tensors. False by default as it causes issues when used with `torch.vmap`."""
    lovely_numpy: bool = False
    """If enabled, will use the lovely-numpy library to format numpy arrays. False by default as it causes some issues with other libaries."""

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
    if config.lovely_tensors:
        try:
            import lovely_tensors  # type: ignore

            lovely_tensors.monkey_patch()
        except ImportError:
            logging.warning(
                "Failed to import `lovely_tensors`. Ignoring pretty PyTorch tensor formatting"
            )

    if config.lovely_numpy:
        try:
            import lovely_numpy  # type: ignore

            lovely_numpy.set_config(repr=lovely_numpy.lovely)
        except ImportError:
            logging.warning(
                "Failed to import `lovely_numpy`. Ignoring pretty numpy array formatting"
            )

    log_handlers: list[logging.Handler] = []
    if config.log_save_dir is not None:
        log_file = config.log_save_dir / "python_log.log"
        log_file.touch(exist_ok=True)
        log_handlers.append(logging.FileHandler(log_file))

    if config.rich:
        try:
            from rich.logging import RichHandler  # type: ignore

            log_handlers.append(RichHandler(rich_tracebacks=config.rich_tracebacks))
        except ImportError:
            logging.warning(
                "Failed to import rich. Falling back to default Python logging."
            )

    logging.basicConfig(
        level=config.log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=log_handlers,
    )
