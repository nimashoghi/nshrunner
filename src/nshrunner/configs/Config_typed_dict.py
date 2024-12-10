from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from nshrunner._runner import Config


__codegen__ = True

# Definitions


class SeedConfig(typ.TypedDict, total=False):
    seed: typ.Required[int]
    """Seed for the random number generator."""

    seed_workers: bool
    """Whether to seed the workers of the dataloader (Only applicable to PyTorch Lightning)."""

    use_lightning: bool
    """Whether to use Lightning's seed_everything function (if available)."""


class SnapshotConfig(typ.TypedDict, total=False):
    snapshot_dir: str
    """The directory to save snapshots to."""

    modules: list[str]
    """Modules to snapshot. Default: `[]`."""

    on_module_not_found: typ.Literal["raise"] | typ.Literal["warn"]
    """What to do when a module is not found. Default: `"warn"`."""

    editable_modules: bool
    """Snapshot all editable modules. Default: `False`."""


# Schema entries
class ConfigTypedDict(typ.TypedDict, total=False):
    working_dir: str | str | None
    """The `working_dir` parameter is a string that represents the directory where the program will save its execution files and logs.
        This is used when submitting the program to a SLURM/LSF cluster or when using the `local_sessions` method.
        If `None`, this will default to the current working directory / `nshrunner`."""

    seed: int | SeedConfig | None
    """Seed configuration for the runner."""

    env: dict[str, str] | None
    """Environment variables to set for the session."""

    snapshot: bool | SnapshotConfig
    """Snapshot configuration for the session."""


@typ.overload
def CreateConfig(**dict: typ.Unpack[ConfigTypedDict]) -> Config: ...


@typ.overload
def CreateConfig(data: ConfigTypedDict | Config, /) -> Config: ...


def CreateConfig(*args, **kwargs):
    from nshrunner._runner import Config

    if not args and kwargs:
        # Called with keyword arguments
        return Config.from_dict(kwargs)
    elif len(args) == 1:
        return Config.from_dict_or_instance(args[0])
    else:
        raise TypeError(
            f"CreateConfig accepts either a ConfigTypedDict, "
            f"keyword arguments, or a Config instance"
        )
