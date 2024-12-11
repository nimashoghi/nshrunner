from __future__ import annotations

import signal
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Literal

import nshconfig as C


class SlurmMailConfig(C.Config):
    """Configuration for email notifications"""

    user: str
    """Email address to receive SLURM notifications"""

    types: Sequence[
        Literal[
            "NONE",
            "BEGIN",
            "END",
            "FAIL",
            "REQUEUE",
            "ALL",
            "INVALID_DEPEND",
            "STAGE_OUT",
            "TIME_LIMIT",
            "TIME_LIMIT_90",
            "TIME_LIMIT_80",
            "TIME_LIMIT_50",
            "ARRAY_TASKS",
        ]
    ] = ["END", "FAIL"]
    """Types of events that should trigger email notifications

    Common values:
    - BEGIN: Job start
    - END: Job completion
    - FAIL: Job failure
    - TIME_LIMIT: Job reached time limit
    - ALL: All events
    """


class SlurmBackendConfig(C.Config):
    """Configuration for the SLURM backbone"""

    name: str = "nshrunner"
    """Name of the job. This will appear in SLURM queue listings"""

    account: str | None = None
    """Account to charge for resource usage. Required by some clusters"""

    partition: str | Sequence[str] | None = None
    """SLURM partition(s) to submit to. Can be a single partition or list of partitions

    Common values:
    - gpu: For GPU jobs
    - cpu: For CPU-only jobs
    - debug: For short test runs
    """

    tasks_per_node: int
    """Number of tasks to run per node"""

    cpus_per_task: int
    """Number of CPUs per task"""

    gpus_per_task: int
    """Number of GPUs required per task. Set to 0 for CPU-only tasks"""

    memory_gb_per_node: int | float | Literal["all"]
    """Memory required in gigabytes per node

    Can be specified as:
    - A number (int/float): Amount of memory in GB
    - "all": Request all available memory on the node
    """

    nodes: int
    """Number of nodes to allocate for the job"""

    time: timedelta
    """Maximum wall time for the job. Job will be terminated after this duration"""

    qos: str | None = None
    """Quality of Service (QoS) level for the job. Controls priority and resource limits"""

    constraint: str | Sequence[str] | None = None
    """Node constraints for job allocation. Can be a single constraint or list of constraints

    These constraints can be features defined by the SLURM administrator that are required for the job.
    Multiple constraints are combined using logical AND.
    """

    # Other existing fields
    output_dir: Path | None = None
    """Directory where SLURM output and error files will be written

    Files will be named using SLURM job variables:
    - %j: Job ID
    - %a: Array task ID (for job arrays)

    If None, `nshrunner` will automatically set the output directory based on the
    provided working directory.
    """

    mail: SlurmMailConfig | None = None
    """Email notification settings. If None, no emails will be sent"""

    timeout_delay: timedelta = timedelta(minutes=2)
    """Duration before job end to send timeout signal, allowing graceful shutdown"""

    timeout_signal: signal.Signals = signal.SIGURG
    """Signal to send when job approaches time limit

    Common values:
    - SIGTERM: Standard termination request
    - SIGINT: Interrupt (like Ctrl+C)
    - SIGUSR1/SIGUSR2: User-defined signals
    """

    exclusive: bool = False
    """If True, request exclusive access to nodes (no sharing with other jobs)"""

    def to_slurm_kwargs(self):
        """Convert SimpleSlurmConfig to full SlurmJobKwargs

        Returns
        -------
        SlurmJobKwargs
            Dictionary of arguments compatible with SLURM sbatch command
        """
        from .._submit.slurm import SlurmJobKwargs

        kwargs: SlurmJobKwargs = {
            "name": self.name,
            "nodes": self.nodes,
            "ntasks_per_node": self.tasks_per_node,
            "cpus_per_task": self.cpus_per_task,
            "memory_mb": (
                0  # Request all memory
                if self.memory_gb_per_node == "all"
                else int(self.memory_gb_per_node * 1024)
            ),
            "time": self.time,
            "timeout_signal": self.timeout_signal,
            "timeout_signal_delay": self.timeout_delay,
            "exclusive": self.exclusive,
        }

        if self.output_dir is not None:
            kwargs["output_file"] = self.output_dir / "%j-%a.out"
            kwargs["error_file"] = self.output_dir / "%j-%a.err"

        if self.account is not None:
            kwargs["account"] = self.account
        if self.partition is not None:
            kwargs["partition"] = self.partition
        if self.qos is not None:
            kwargs["qos"] = self.qos
        if self.constraint is not None:
            kwargs["constraint"] = self.constraint

        if self.gpus_per_task > 0:
            kwargs["gpus_per_task"] = self.gpus_per_task

        if self.mail is not None:
            kwargs["mail_user"] = self.mail.user
            kwargs["mail_type"] = self.mail.types

        return kwargs
