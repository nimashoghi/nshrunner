from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from nshrunner.backends.screen import ScreenBackendConfig


__codegen__ = True

"""Configuration for the GNU screen backbone."""

# Definitions


class LoggingConfig(typ.TypedDict, total=False):
    """Configuration for screen session logging."""

    screen_log_file: str | None
    """Path to save screen session log. Uses screen's -L flag."""

    output_file: str | None
    """Path to save stdout from the command."""

    error_file: str | None
    """Path to save stderr from the command."""


# Schema entries
class ScreenBackendConfigTypedDict(typ.TypedDict, total=False):
    """Configuration for the GNU screen backbone."""

    name: str
    """Name of the screen session. Used to identify and reattach to sessions."""

    logging: LoggingConfig
    """Logging configuration for the screen session and commands."""

    attach: bool
    """If True, attach to the screen session after creating it
    
    - True: Stay attached to the session (interactive)
    - False: Detach after starting (daemon mode)."""

    pause_before_exit: bool
    """If True, wait for user input before closing the screen session
    
    Useful for examining output before the session closes."""

    setup_commands: list[str] | None
    """Commands to run before the main command
    
    These are executed in the same environment as the main command."""

    environment: dict[str, str] | None
    """Environment variables to set for the session
    
    These will be exported before running any commands."""

    command_prefix: str | None
    """Optional prefix to add before the main command
    
    Example: "python -u" to run Python with unbuffered output."""

    emit_metadata: bool
    """If True, save job submission metadata to JSON files."""


@typ.overload
def CreateScreenBackendConfig(
    **dict: typ.Unpack[ScreenBackendConfigTypedDict],
) -> ScreenBackendConfig: ...


@typ.overload
def CreateScreenBackendConfig(
    data: ScreenBackendConfigTypedDict | ScreenBackendConfig, /
) -> ScreenBackendConfig: ...


def CreateScreenBackendConfig(*args, **kwargs):
    from nshrunner.backends.screen import ScreenBackendConfig

    if not args and kwargs:
        # Called with keyword arguments
        return ScreenBackendConfig.from_dict(kwargs)
    elif len(args) == 1:
        return ScreenBackendConfig.from_dict_or_instance(args[0])
    else:
        raise TypeError(
            f"CreateScreenBackendConfig accepts either a ScreenBackendConfigTypedDict, "
            f"keyword arguments, or a ScreenBackendConfig instance"
        )
