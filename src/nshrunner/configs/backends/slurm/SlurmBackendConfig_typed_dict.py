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


# Schema entries
class SlurmBackendConfigTypedDict(typ.TypedDict, total=False):
    """Configuration for the SLURM backbone."""

    name: str
    """Name of the job. This will appear in SLURM queue listings."""

    account: str | None
    """Account to charge for resource usage. Required by some clusters."""

    partition: str | list[str] | None
    """SLURM partition(s) to submit to. Can be a single partition or list of partitions
    
    Common values:
    - gpu: For GPU jobs
    - cpu: For CPU-only jobs
    - debug: For short test runs."""

    tasks_per_node: typ.Required[int]
    """Number of tasks to run per node."""

    cpus_per_task: typ.Required[int]
    """Number of CPUs per task."""

    gpus_per_task: typ.Required[int]
    """Number of GPUs required per task. Set to 0 for CPU-only tasks."""

    memory_gb_per_node: typ.Required[int | float | str]
    """Memory required in gigabytes per node
    
    Can be specified as:
    - A number (int/float): Amount of memory in GB
    - "all": Request all available memory on the node."""

    nodes: typ.Required[int]
    """Number of nodes to allocate for the job."""

    time: typ.Required[str]
    """Maximum wall time for the job. Job will be terminated after this duration."""

    qos: str | None
    """Quality of Service (QoS) level for the job. Controls priority and resource limits."""

    constraint: str | list[str] | None
    """Node constraints for job allocation. Can be a single constraint or list of constraints
    
    These constraints can be features defined by the SLURM administrator that are required for the job.
    Multiple constraints are combined using logical AND."""

    output_dir: str | None
    """Directory where SLURM output and error files will be written
    
    Files will be named using SLURM job variables:
    - %j: Job ID
    - %a: Array task ID (for job arrays)
    
    If None, `nshrunner` will automatically set the output directory based on the
    provided working directory."""

    mail: SlurmMailConfig | None
    """Email notification settings. If None, no emails will be sent."""

    timeout_delay: str
    """Duration before job end to send timeout signal, allowing graceful shutdown."""

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
