from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from nshrunner._logging import PythonLoggingConfig


__codegen__ = True

# Definitions
LogLevel = typ.TypeAliasType(
    "LogLevel",
    typ.Literal["CRITICAL"]
    | typ.Literal["FATAL"]
    | typ.Literal["ERROR"]
    | typ.Literal["WARN"]
    | typ.Literal["WARNING"]
    | typ.Literal["INFO"]
    | typ.Literal["DEBUG"],
)


# Schema entries
class PythonLoggingConfigTypedDict(typ.TypedDict, total=False):
    log_level: LogLevel
    """Log level to use for the Python logger."""

    log_save_dir: str | None
    """Directory to save logs to. If None, logs will not be saved."""

    treescope: bool
    """If enabled, will use the treescope library to visualize data structures in notebooks."""

    treescope_autovisualize_arrays: bool
    """If enabled, will automatically visualize arrays with treescope (if `treescope` is enabled)."""

    lovely_tensors: bool
    """If enabled, will use the lovely-tensors library to format PyTorch tensors. False by default as it causes issues when used with `torch.vmap`."""

    lovely_numpy: bool
    """If enabled, will use the lovely-numpy library to format numpy arrays. False by default as it causes some issues with other libaries."""

    rich: bool
    """If enabled, will use the rich library to format the Python logger output."""

    rich_tracebacks: bool
    """If enabled, will use the rich library to format the Python logger tracebacks."""


@typ.overload
def CreatePythonLoggingConfig(
    dict: PythonLoggingConfigTypedDict, /
) -> PythonLoggingConfig: ...


@typ.overload
def CreatePythonLoggingConfig(
    **dict: typ.Unpack[PythonLoggingConfigTypedDict],
) -> PythonLoggingConfig: ...


def CreatePythonLoggingConfig(*args, **kwargs):
    from nshrunner._logging import PythonLoggingConfig

    dict = args[0] if args else kwargs
    return PythonLoggingConfig.model_validate(dict)
