import copy
import logging
import os
import signal
import subprocess
from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

from typing_extensions import (
    TypeAlias,
    TypedDict,
    TypeVar,
    TypeVarTuple,
    Unpack,
    assert_never,
)

from . import lsf, slurm
from ._output import SubmitOutput

TArgs = TypeVarTuple("TArgs")
_Path: TypeAlias = str | Path | os.PathLike

log = logging.getLogger(__name__)


class GenericJobKwargs(TypedDict, total=False):
    name: str
    """The name of the job."""

    partition: str | Sequence[str]
    """The partition or queue to submit the job to. Same as `queue`."""

    queue: str | Sequence[str]
    """The queue to submit the job to. Same as `partition`."""

    qos: str
    """
    The quality of service to submit the job to.

    This corresponds to the "--qos" option in sbatch (only for Slurm).
    """

    account: str
    """The account (or project) to charge the job to. Same as `project`."""

    project: str
    """The project (or account) to charge the job to. Same as `account`."""

    output_file: _Path
    """
    The file to write the job output to.

    This corresponds to the "-o" option in bsub. If not specified, the output will be written to the default output file.
    """

    error_file: _Path
    """
    The file to write the job errors to.

    This corresponds to the "-e" option in bsub. If not specified, the errors will be written to the default error file.
    """

    nodes: int
    """The number of nodes to request."""

    tasks_per_node: int
    """The number of tasks to request per node."""

    cpus_per_task: int
    """The number of CPUs to request per task."""

    gpus_per_task: int
    """The number of GPUs to request per task."""

    memory_mb: int
    """The maximum memory for the job in MB."""

    walltime: timedelta
    """The maximum walltime for the job."""

    email: str
    """The email address to send notifications to."""

    notifications: set[Literal["begin", "end"]]
    """The notifications to send via email."""

    setup_commands: Sequence[str]
    """
    The setup commands to run before the job.

    These commands will be executed prior to everything else in the job script.
    """

    environment: Mapping[str, str]
    """
    The environment variables to set for the job.

    These variables will be set prior to executing any commands in the job script.
    """

    command_prefix: str
    """
    A command to prefix the job command with.

    This is used to add commands like `srun` or `jsrun` to the job command.
    """

    constraint: str | Sequence[str]
    """
    The constraint to request for the job. For SLRUM, this corresponds to the `--constraint` option. For LSF, this is unused.
    """

    signal: signal.Signals
    """The signal that will be sent to the job when it is time to stop it."""

    command_template: str
    """
    The template for the command to execute the helper script.

    Default: `bash {script}`.
    """

    requeue_on_preempt: bool
    """
    Whether to requeue the job if it is preempted.

    This corresponds to the "--requeue" option in sbatch (only for Slurm).
    """

    slurm_options: slurm.SlurmJobKwargs
    """Additional keyword arguments for Slurm jobs."""

    lsf_options: lsf.LSFJobKwargs
    """Additional keyword arguments for LSF jobs."""


Scheduler: TypeAlias = Literal["slurm", "lsf"]


T = TypeVar("T", infer_variance=True)


def _one_of(*fns: Callable[[], T | None]) -> T | None:
    values = [value for fn in fns if (value := fn()) is not None]

    # Only one (or zero) value should be set. If not, raise an error.
    if len(set(values)) > 1:
        raise ValueError(f"Multiple values set: {values}")

    return next((value for value in values if value is not None), None)


def _to_slurm(kwargs: GenericJobKwargs) -> slurm.SlurmJobKwargs:
    slurm_kwargs: slurm.SlurmJobKwargs = {}
    if (name := kwargs.get("name")) is not None:
        slurm_kwargs["name"] = name
    if (
        account := _one_of(
            lambda: kwargs.get("account"),
            lambda: kwargs.get("project"),
        )
    ) is not None:
        slurm_kwargs["account"] = account
    if (
        partition := _one_of(
            lambda: kwargs.get("partition"),
            lambda: kwargs.get("queue"),
        )
    ) is not None:
        slurm_kwargs["partition"] = partition
    if (qos := kwargs.get("qos")) is not None:
        slurm_kwargs["qos"] = qos
    if (output_file := kwargs.get("output_file")) is not None:
        slurm_kwargs["output_file"] = output_file
    if (error_file := kwargs.get("error_file")) is not None:
        slurm_kwargs["error_file"] = error_file
    if (walltime := kwargs.get("walltime")) is not None:
        slurm_kwargs["time"] = walltime
    if (memory_mb := kwargs.get("memory_mb")) is not None:
        slurm_kwargs["memory_mb"] = memory_mb
    if (nodes := kwargs.get("nodes")) is not None:
        slurm_kwargs["nodes"] = nodes
    if (tasks_per_node := kwargs.get("tasks_per_node")) is not None:
        slurm_kwargs["ntasks_per_node"] = tasks_per_node
    if (cpus_per_task := kwargs.get("cpus_per_task")) is not None:
        slurm_kwargs["cpus_per_task"] = cpus_per_task
    if (gpus_per_task := kwargs.get("gpus_per_task")) is not None:
        slurm_kwargs["gpus_per_task"] = gpus_per_task
    if (constraint := kwargs.get("constraint")) is not None:
        slurm_kwargs["constraint"] = constraint
    if (signal := kwargs.get("signal")) is not None:
        slurm_kwargs["signal"] = signal
    if (email := kwargs.get("email")) is not None:
        slurm_kwargs["mail_user"] = email
    if (notifications := kwargs.get("notifications")) is not None:
        mail_type: list[slurm.MailType] = []
        for notification in notifications:
            match notification:
                case "begin":
                    mail_type.append("BEGIN")
                case "end":
                    mail_type.append("END")
                case _:
                    raise ValueError(f"Unknown notification type: {notification}")
        slurm_kwargs["mail_type"] = mail_type
    if (setup_commands := kwargs.get("setup_commands")) is not None:
        slurm_kwargs["setup_commands"] = setup_commands
    if (environment := kwargs.get("environment")) is not None:
        slurm_kwargs["environment"] = environment
    if (command_prefix := kwargs.get("command_prefix")) is not None:
        slurm_kwargs["command_prefix"] = command_prefix
    if (requeue_on_preempt := kwargs.get("requeue_on_preempt")) is not None:
        slurm_kwargs["requeue"] = requeue_on_preempt
    if (additional_kwargs := kwargs.get("slurm_options")) is not None:
        slurm_kwargs.update(additional_kwargs)

    return slurm_kwargs


def _to_lsf(kwargs: GenericJobKwargs) -> lsf.LSFJobKwargs:
    lsf_kwargs: lsf.LSFJobKwargs = {}
    if (name := kwargs.get("name")) is not None:
        lsf_kwargs["name"] = name
    if (
        account := _one_of(
            lambda: kwargs.get("account"),
            lambda: kwargs.get("project"),
        )
    ) is not None:
        lsf_kwargs["project"] = account
    if (
        partition := _one_of(
            lambda: kwargs.get("partition"),
            lambda: kwargs.get("queue"),
        )
    ) is not None:
        lsf_kwargs["queue"] = partition
    if (output_file := kwargs.get("output_file")) is not None:
        lsf_kwargs["output_file"] = output_file
    if (error_file := kwargs.get("error_file")) is not None:
        lsf_kwargs["error_file"] = error_file
    if (walltime := kwargs.get("walltime")) is not None:
        lsf_kwargs["walltime"] = walltime
    if (memory_mb := kwargs.get("memory_mb")) is not None:
        lsf_kwargs["memory_mb"] = memory_mb
    if (nodes := kwargs.get("nodes")) is not None:
        lsf_kwargs["nodes"] = nodes
    if (tasks_per_node := kwargs.get("tasks_per_node")) is not None:
        lsf_kwargs["rs_per_node"] = tasks_per_node
    if (cpus_per_task := kwargs.get("cpus_per_task")) is not None:
        lsf_kwargs["cpus_per_rs"] = cpus_per_task
    if (gpus_per_task := kwargs.get("gpus_per_task")) is not None:
        lsf_kwargs["gpus_per_rs"] = gpus_per_task
    if (constraint := kwargs.get("constraint")) is not None:
        log.warning(f'LSF does not support constraints, ignoring "{constraint=}".')
    if (email := kwargs.get("email")) is not None:
        lsf_kwargs["email"] = email
    if (notifications := kwargs.get("notifications")) is not None:
        if "begin" in notifications:
            lsf_kwargs["notify_begin"] = True
        if "end" in notifications:
            lsf_kwargs["notify_end"] = True
    if (setup_commands := kwargs.get("setup_commands")) is not None:
        lsf_kwargs["setup_commands"] = setup_commands
    if (environment := kwargs.get("environment")) is not None:
        lsf_kwargs["environment"] = environment
    if (command_prefix := kwargs.get("command_prefix")) is not None:
        lsf_kwargs["command_prefix"] = command_prefix
    if (signal := kwargs.get("signal")) is not None:
        lsf_kwargs["signal"] = signal
    if (requeue_on_preempt := kwargs.get("requeue_on_preempt")) is not None:
        log.warning(
            f'LSF does not support requeueing, ignoring "{requeue_on_preempt=}".'
        )
    if (additional_kwargs := kwargs.get("lsf_options")) is not None:
        lsf_kwargs.update(additional_kwargs)

    return lsf_kwargs


def validate_kwargs(scheduler: Scheduler, kwargs: GenericJobKwargs) -> None:
    match scheduler:
        case "slurm":
            _to_slurm(copy.deepcopy(kwargs))
        case "lsf":
            _to_lsf(copy.deepcopy(kwargs))
        case _:
            assert_never(scheduler)


def to_array_batch_script(
    scheduler: Scheduler,
    dest: Path,
    callable: Callable[[Unpack[TArgs]], Any],
    args_list: Sequence[tuple[Unpack[TArgs]]],
    /,
    job_index_variable: str | None = None,
    print_environment_info: bool = False,
    python_command_prefix: str | None = None,
    **kwargs: Unpack[GenericJobKwargs],
) -> SubmitOutput:
    job_index_variable_kwargs = {}
    if job_index_variable is not None:
        job_index_variable_kwargs["job_index_variable"] = job_index_variable
    match scheduler:
        case "slurm":
            slurm_kwargs = _to_slurm(kwargs)
            return slurm.to_array_batch_script(
                dest,
                callable,
                args_list,
                **job_index_variable_kwargs,
                print_environment_info=print_environment_info,
                python_command_prefix=python_command_prefix,
                **slurm_kwargs,
            )
        case "lsf":
            lsf_kwargs = _to_lsf(kwargs)
            return lsf.to_array_batch_script(
                dest,
                callable,
                args_list,
                **job_index_variable_kwargs,
                print_environment_info=print_environment_info,
                python_command_prefix=python_command_prefix,
                **lsf_kwargs,
            )
        case _:
            assert_never(scheduler)


def infer_current_scheduler() -> Scheduler:
    # First, we check for `bsub` as it's much less common than `sbatch`.
    try:
        subprocess.check_output(["bsub", "-V"])
        return "lsf"
    except BaseException:
        pass

    # Next, we check for `sbatch` as it's the most common scheduler.
    try:
        subprocess.check_output(["sbatch", "--version"])
        return "slurm"
    except BaseException:
        pass

    raise RuntimeError("Could not determine the current scheduler.")
