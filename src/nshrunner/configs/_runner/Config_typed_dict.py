from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from nshrunner._runner import Config


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


class PythonLoggingConfig(typ.TypedDict, total=False):
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


class SeedConfig(typ.TypedDict, total=False):
    seed: typ.Required[int]
    """Seed for the random number generator."""

    seed_workers: bool
    """Whether to seed the workers of the dataloader (Only applicable to PyTorch Lightning)."""

    use_lightning: bool
    """Whether to use Lightning's seed_everything function (if available)."""


Path = typ.TypeAliasType("Path", str | str)


# Schema entries
class ConfigTypedDict(typ.TypedDict, total=False):
    working_dir: Path | None
    """The `working_dir` parameter is a string that represents the directory where the program will save its execution files and logs.
        This is used when submitting the program to a SLURM/LSF cluster or when using the `local_sessions` method.
        If `None`, this will default to the current working directory / `nshrunner`."""

    python_logging: PythonLoggingConfig | None
    """Logging configuration for the runner."""

    seed: SeedConfig | None
    """Seed configuration for the runner."""

    env: dict[str, str] | None
    """Environment variables to set for the session."""

    auto_snapshot_args_resolved_modules: bool
    """If enabled, `nshsnap` will automatically look through the function
    arguments and snapshot any third-party modules that are resolved."""

    auto_snapshot_editable: bool
    """If enabled, `nshsnap` will automatically snapshot any editable packages."""


@typ.overload
def CreateConfig(dict: ConfigTypedDict, /) -> Config: ...


@typ.overload
def CreateConfig(**dict: typ.Unpack[ConfigTypedDict]) -> Config: ...


def CreateConfig(*args, **kwargs):
    from nshrunner._runner import Config

    dict = args[0] if args else kwargs
    return Config.model_validate(dict)
