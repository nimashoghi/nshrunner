from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from nshrunner.backends.slurm import SlurmBackendConfig


__codegen__ = True

"""Configuration for the SLURM backbone."""

# Definitions
Signals = typ.TypeAliasType(
    "Signals",
    typ.Literal[1]
    | typ.Literal[2]
    | typ.Literal[3]
    | typ.Literal[4]
    | typ.Literal[5]
    | typ.Literal[6]
    | typ.Literal[6]
    | typ.Literal[7]
    | typ.Literal[8]
    | typ.Literal[9]
    | typ.Literal[10]
    | typ.Literal[11]
    | typ.Literal[12]
    | typ.Literal[13]
    | typ.Literal[14]
    | typ.Literal[15]
    | typ.Literal[16]
    | typ.Literal[17]
    | typ.Literal[17]
    | typ.Literal[18]
    | typ.Literal[19]
    | typ.Literal[20]
    | typ.Literal[21]
    | typ.Literal[22]
    | typ.Literal[23]
    | typ.Literal[24]
    | typ.Literal[25]
    | typ.Literal[26]
    | typ.Literal[27]
    | typ.Literal[28]
    | typ.Literal[29]
    | typ.Literal[29]
    | typ.Literal[30]
    | typ.Literal[31]
    | typ.Literal[34]
    | typ.Literal[64],
)


class SlurmMailConfig(typ.TypedDict, total=False):
    """Configuration for email notifications."""

    user: typ.Required[str]
    """Email address to receive SLURM notifications."""

    types: list[
        typ.Literal["NONE"]
        | typ.Literal["BEGIN"]
        | typ.Literal["END"]
        | typ.Literal["FAIL"]
        | typ.Literal["REQUEUE"]
        | typ.Literal["ALL"]
        | typ.Literal["INVALID_DEPEND"]
        | typ.Literal["STAGE_OUT"]
        | typ.Literal["TIME_LIMIT"]
        | typ.Literal["TIME_LIMIT_90"]
        | typ.Literal["TIME_LIMIT_80"]
        | typ.Literal["TIME_LIMIT_50"]
        | typ.Literal["ARRAY_TASKS"]
    ]
    """Types of events that should trigger email notifications
    
    Common values:
    - BEGIN: Job start
    - END: Job completion
    - FAIL: Job failure
    - TIME_LIMIT: Job reached time limit
    - ALL: All events."""


class SlurmResourcesConfig(typ.TypedDict):
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


# Schema entries
class SlurmBackendConfigTypedDict(typ.TypedDict, total=False):
    """Configuration for the SLURM backbone."""

    name: typ.Required[str]
    """Name of the job. This will appear in SLURM queue listings."""

    account: str | None
    """Account to charge for resource usage. Required by some clusters."""

    partition: str | list[str] | None
    """SLURM partition(s) to submit to. Can be a single partition or list of partitions
    
    Common values:
    - gpu: For GPU jobs
    - cpu: For CPU-only jobs
    - debug: For short test runs."""

    qos: str | None
    """Quality of Service (QoS) level for the job. Controls priority and resource limits."""

    resources: typ.Required[SlurmResourcesConfig]
    """Resource requirements for the job including CPU, GPU, memory, and time limits."""

    output_dir: typ.Required[str]
    """Directory where SLURM output and error files will be written
    
    Files will be named using SLURM job variables:
    - %j: Job ID
    - %a: Array task ID (for job arrays)."""

    mail: SlurmMailConfig | None
    """Email notification settings. If None, no emails will be sent."""

    timeout_min: int
    """Minutes before job end to send timeout signal, allowing graceful shutdown."""

    timeout_signal: Signals
    """Signal to send when job approaches time limit
    
    Common values:
    - SIGTERM: Standard termination request
    - SIGINT: Interrupt (like Ctrl+C)
    - SIGUSR1/SIGUSR2: User-defined signals."""

    exclusive: bool
    """If True, request exclusive access to nodes (no sharing with other jobs)."""


@typ.overload
def CreateSlurmBackendConfig(
    **dict: typ.Unpack[SlurmBackendConfigTypedDict],
) -> SlurmBackendConfig: ...


@typ.overload
def CreateSlurmBackendConfig(
    data: SlurmBackendConfigTypedDict | SlurmBackendConfig, /
) -> SlurmBackendConfig: ...


def CreateSlurmBackendConfig(*args, **kwargs):
    from nshrunner.backends.slurm import SlurmBackendConfig

    if not args and kwargs:
        # Called with keyword arguments
        return SlurmBackendConfig.from_dict(kwargs)
    elif len(args) == 1:
        return SlurmBackendConfig.from_dict_or_instance(args[0])
    else:
        raise TypeError(
            f"CreateSlurmBackendConfig accepts either a SlurmBackendConfigTypedDict, "
            f"keyword arguments, or a SlurmBackendConfig instance"
        )
