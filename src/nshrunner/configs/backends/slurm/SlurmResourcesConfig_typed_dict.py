from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from nshrunner.backends.slurm import SlurmResourcesConfig


__codegen__ = True

"""Configuration for computational resources."""


# Schema entries
class SlurmResourcesConfigTypedDict(typ.TypedDict):
    """Configuration for computational resources."""

    cpus: int
    """Number of CPUs per task."""

    gpus: int
    """Number of GPUs required per node. Set to 0 for CPU-only jobs."""

    memory_gb: float
    """Memory required in gigabytes."""

    nodes: int
    """Number of nodes to allocate for the job."""

    time: str
    """Maximum wall time for the job. Job will be terminated after this duration."""


@typ.overload
def CreateSlurmResourcesConfig(
    **dict: typ.Unpack[SlurmResourcesConfigTypedDict],
) -> SlurmResourcesConfig: ...


@typ.overload
def CreateSlurmResourcesConfig(
    data: SlurmResourcesConfigTypedDict | SlurmResourcesConfig, /
) -> SlurmResourcesConfig: ...


def CreateSlurmResourcesConfig(*args, **kwargs):
    from nshrunner.backends.slurm import SlurmResourcesConfig

    if not args and kwargs:
        # Called with keyword arguments
        return SlurmResourcesConfig.from_dict(kwargs)
    elif len(args) == 1:
        return SlurmResourcesConfig.from_dict_or_instance(args[0])
    else:
        raise TypeError(
            f"CreateSlurmResourcesConfig accepts either a SlurmResourcesConfigTypedDict, "
            f"keyword arguments, or a SlurmResourcesConfig instance"
        )
