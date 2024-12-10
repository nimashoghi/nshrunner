from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import nshconfig as C


class LoggingConfig(C.Config):
    """Configuration for screen session logging"""

    screen_log_file: Path | None = None
    """Path to save screen session log. Uses screen's -L flag"""

    output_file: Path | None = None
    """Path to save stdout from the command"""

    error_file: Path | None = None
    """Path to save stderr from the command"""


class ScreenBackendConfig(C.Config):
    """Configuration for the GNU screen backbone"""

    name: str = "nshrunner"
    """Name of the screen session. Used to identify and reattach to sessions"""

    logging: LoggingConfig = LoggingConfig()
    """Logging configuration for the screen session and commands"""

    attach: bool = True
    """If True, attach to the screen session after creating it

    - True: Stay attached to the session (interactive)
    - False: Detach after starting (daemon mode)
    """

    pause_before_exit: bool = True
    """If True, wait for user input before closing the screen session

    Useful for examining output before the session closes
    """

    setup_commands: Sequence[str] | None = None
    """Commands to run before the main command

    These are executed in the same environment as the main command"""

    environment: dict[str, str] | None = None
    """Environment variables to set for the session

    These will be exported before running any commands"""

    command_prefix: str | None = None
    """Optional prefix to add before the main command

    Example: "python -u" to run Python with unbuffered output"""

    emit_metadata: bool = True
    """If True, save job submission metadata to JSON files"""

    def to_screen_kwargs(self):
        """Convert SimpleScreenConfig to full ScreenJobKwargs

        Returns
        -------
        ScreenJobKwargs
            Dictionary of arguments compatible with screen command
        """
        from .._submit.screen import ScreenJobKwargs

        kwargs: ScreenJobKwargs = {
            "name": self.name,
            "attach": self.attach,
            "pause_before_exit": self.pause_before_exit,
            "emit_metadata": self.emit_metadata,
        }

        if self.logging.screen_log_file is not None:
            kwargs["screen_log_file"] = self.logging.screen_log_file
        if self.logging.output_file is not None:
            kwargs["output_file"] = self.logging.output_file
        if self.logging.error_file is not None:
            kwargs["error_file"] = self.logging.error_file

        if self.setup_commands is not None:
            kwargs["setup_commands"] = self.setup_commands

        if self.environment is not None:
            kwargs["environment"] = self.environment

        if self.command_prefix is not None:
            kwargs["command_prefix"] = self.command_prefix

        return kwargs
