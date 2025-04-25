from __future__ import annotations

import nshsnap._config
import nshsnap._snapshot
import typing_extensions as typ

import nshrunner._seed

if typ.TYPE_CHECKING:
    from nshrunner._config import Config


__codegen__ = True

# Definitions
ActiveSnapshot = typ.TypeAliasType("ActiveSnapshot", nshsnap._snapshot.ActiveSnapshot)


class SeedConfigTypedDict(typ.TypedDict, total=False):
    seed: typ.Required[int]
    """Seed for the random number generator."""

    seed_workers: bool
    """Whether to seed the workers of the dataloader (Only applicable to PyTorch Lightning)."""

    use_lightning: bool
    """Whether to use Lightning's seed_everything function (if available)."""


SeedConfig = typ.TypeAliasType(
    "SeedConfig", SeedConfigTypedDict | nshrunner._seed.SeedConfig
)


class SnapshotConfigTypedDict(typ.TypedDict, total=False):
    snapshot_dir: str | None
    """The directory to save snapshots to."""

    modules: list[str]
    """Modules to snapshot. Default: `[]`."""

    on_module_not_found: typ.Literal["raise"] | typ.Literal["warn"]
    """What to do when a module is not found. Default: `"warn"`."""

    editable_modules: bool
    """Snapshot all editable modules. Default: `True`."""


SnapshotConfig = typ.TypeAliasType(
    "SnapshotConfig", SnapshotConfigTypedDict | nshsnap._config.SnapshotConfig
)
SnapshotModuleInfo = typ.TypeAliasType(
    "SnapshotModuleInfo", nshsnap._snapshot.SnapshotModuleInfo
)


# Schema entries
class ConfigTypedDict(typ.TypedDict, total=False):
    working_dir: (
        str | typ.Literal["cwd"] | typ.Literal["tmp"] | typ.Literal["home-cache"]
    )
    """The `working_dir` parameter is a string that represents the directory where the program will save its execution files and logs.
    This is used when submitting the program to a SLURM/LSF cluster or when using the `local_sessions` method.
    
    Accepted values are:
    - "cwd": The current working directory.
    - "tmp": The temporary directory.
    - "home-cache" (default): The cache directory in the user's home directory (i.e., `~/.cache/nshrunner`)."""

    seed: int | SeedConfig | None
    """Seed configuration for the runner."""

    env: dict[str, str] | None
    """Environment variables to set for the session."""

    snapshot: bool | SnapshotConfig | ActiveSnapshot | None
    """Snapshot configuration for the session.
    
    If `True`, a snapshot will be created with the default configuration.
    If `False` or `None`, no snapshot will be created.
    If a `SnapshotConfig` object is provided, it will be used to configure the snapshot.
    If a `ActiveSnapshot` object is provided, it will re-use the existing snapshot from the given snapshot info."""

    save_main_script: bool
    """Whether to save the main script or notebook that's being executed."""

    save_git_diff: bool
    """Whether to save the git diff if the current directory is in a git repository."""


@typ.overload
def CreateConfig(**dict: typ.Unpack[ConfigTypedDict]) -> Config: ...


@typ.overload
def CreateConfig(data: ConfigTypedDict | Config, /) -> Config: ...


def CreateConfig(*args, **kwargs):
    from nshrunner._config import Config

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
