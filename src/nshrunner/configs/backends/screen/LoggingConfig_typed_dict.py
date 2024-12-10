from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from nshrunner.backends.screen import LoggingConfig


__codegen__ = True

"""Configuration for screen session logging."""


# Schema entries
class LoggingConfigTypedDict(typ.TypedDict, total=False):
    """Configuration for screen session logging."""

    screen_log_file: str | None
    """Path to save screen session log. Uses screen's -L flag."""

    output_file: str | None
    """Path to save stdout from the command."""

    error_file: str | None
    """Path to save stderr from the command."""


@typ.overload
def CreateLoggingConfig(
    **dict: typ.Unpack[LoggingConfigTypedDict],
) -> LoggingConfig: ...


@typ.overload
def CreateLoggingConfig(
    data: LoggingConfigTypedDict | LoggingConfig, /
) -> LoggingConfig: ...


def CreateLoggingConfig(*args, **kwargs):
    from nshrunner.backends.screen import LoggingConfig

    if not args and kwargs:
        # Called with keyword arguments
        return LoggingConfig.from_dict(kwargs)
    elif len(args) == 1:
        return LoggingConfig.from_dict_or_instance(args[0])
    else:
        raise TypeError(
            f"CreateLoggingConfig accepts either a LoggingConfigTypedDict, "
            f"keyword arguments, or a LoggingConfig instance"
        )
