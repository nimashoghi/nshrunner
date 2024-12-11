from __future__ import annotations

import copy
import logging
import math
import os
import signal
from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Literal, cast

from deepmerge import always_merger
from typing_extensions import TypeAliasType, TypedDict

from .. import _env
from ._util import (
    Submission,
    emit_on_exit_commands,
    set_default_envs,
    write_run_metadata_commands,
    write_submission_meta,
)

log = logging.getLogger(__name__)

_Path = TypeAliasType("_Path", str | Path | os.PathLike)
MailType = TypeAliasType(
    "MailType",
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
    ],
)


class SlurmJobKwargs(TypedDict, total=False):
    name: str
    """
    The name of the job.

    This corresponds to the "-J" option in sbatch.
    """

    account: str
    """
    The account to charge resources used by this job to.

    This corresponds to the "-A" option in sbatch.
    """

    partition: str | Sequence[str]
    """
    The partition to submit the job to.

    This corresponds to the "-p" option in sbatch. If not specified, the default partition will be used.
    Multiple partitions can be specified, and they will be combined using logical OR.
    """

    qos: str
    """
    The quality of service to submit the job to.

    This corresponds to the "--qos" option in sbatch.
    """

    output_file: _Path
    """
    The file to write the job output to.

    This corresponds to the "-o" option in sbatch. If not specified, the output will be written to the default output file.
    """

    error_file: _Path
    """
    The file to write the job errors to.

    This corresponds to the "-e" option in sbatch. If not specified, the errors will be written to the default error file.
    """

    time: timedelta | Literal[0]
    """
    The maximum time for the job.

    This corresponds to the "-t" option in sbatch. A value of 0 means no time limit.
    """

    memory_mb: int
    """
    The maximum memory for the job in MB.

    This corresponds to the "--mem" option in sbatch. If not specified, the default memory limit will be used.
    """

    memory_per_cpu_mb: int
    """
    The minimum memory required per usable allocated CPU.

    This corresponds to the "--mem-per-cpu" option in sbatch. If not specified, the default memory limit will be used.
    """

    memory_per_gpu_mb: int
    """
    The minimum memory required per allocated GPU.

    This corresponds to the "--mem-per-gpu" option in sbatch. If not specified, the default memory limit will be used.
    """

    cpus_per_task: int
    """
    Advise the Slurm controller that ensuing job steps will require _ncpus_ number of processors per task.

    This corresponds to the "-c" option in sbatch.
    """

    nodes: int
    """
    The number of nodes to use for the job.

    This corresponds to the "-N" option in sbatch. The default is 1 node.
    """

    ntasks: int
    """
    The number of tasks to use for the job.

    This corresponds to the "-n" option in sbatch. The default is 1 task.
    """

    ntasks_per_node: int
    """
    The number of tasks for each node.

    This corresponds to the "--ntasks-per-node" option in sbatch.
    """

    constraint: str | Sequence[str]
    """
    Nodes can have features assigned to them by the Slurm administrator. Users can specify which of these features are required by their job using the constraint option.

    This corresponds to the "-C" option in sbatch.
    """

    gres: str | Sequence[str]
    """
    Specifies a comma-delimited list of generic consumable resources.

    This corresponds to the "--gres" option in sbatch.
    """

    gpus: int | str
    """
    Specify the total number of GPUs required for the job. An optional GPU type specification can be supplied.

    This corresponds to the "-G" option in sbatch.
    """

    gpus_per_node: int | str
    """
    Specify the number of GPUs required for the job on each node included in the job's resource allocation. An optional GPU type specification can be supplied.

    This corresponds to the "--gpus-per-node" option in sbatch.
    """

    gpus_per_task: int
    """
    Specify the number of GPUs required for the job on each task to be spawned in the job's resource allocation. An optional GPU type specification can be supplied.

    This corresponds to the "--gpus-per-task" option in sbatch.
    """

    mail_user: str
    """
    User to receive email notification of state changes as defined by mail_type.

    This corresponds to the "--mail-user" option in sbatch.
    """

    mail_type: MailType | Sequence[MailType]
    """
    Notify user by email when certain event types occur.

    This corresponds to the "--mail-type" option in sbatch.
    """

    dependency: str
    """
    Defer the start of this job until the specified dependencies have been satisfied.

    This corresponds to the "-d" option in sbatch.
    """

    exclusive: bool
    """
    The job allocation can not share nodes with other running jobs.

    This corresponds to the "--exclusive" option in sbatch.
    """

    timeout_signal: signal.Signals
    """
    The signal to send to the job when the job is being terminated.

    This corresponds to the "--signal" option in sbatch.
    """

    timeout_signal_delay: timedelta
    """
    The delay before sending the signal to the job.

    This corresponds to the "--signal ...@[delay]" option in sbatch.
    """

    preempt_signal: signal.Signals
    """
    The signal to send to the job when it is preempted.
    """

    open_mode: str
    """
    The open mode for the output and error files.

    This corresponds to the "--open-mode" option in sbatch.

    Valid values are "append" and "truncate".
    """

    requeue: bool
    """
    Requeues the job if it's pre-empted.

    This corresponds to the "--requeue" option in sbatch.
    """

    setup_commands: Sequence[str]
    """
    The setup commands to run before the job.

    These commands will be executed prior to everything else in the job script.
    They will be included in both the worker script (the one that runs inside of srun/jsrun)
    and not the submission script (the one that is submitted to the job scheduler).
    """

    submission_script_setup_commands: Sequence[str]
    """
    Same as `setup_commands`, but only for the submission script (and not
    the worker scripts executed by srun/jsrun).
    """

    environment: Mapping[str, str]
    """
    The environment variables to set for the job.

    These variables will be set prior to executing any commands in the job script.
    """

    command_prefix: str
    """
    A command to prefix the job command with.

    This is used to add commands like `srun` to the job command.
    """

    srun_flags: str | Sequence[str]
    """
    The flags to pass to the `srun` command.
    """

    emit_metadata: bool
    """
    Whether to emit metadata about the job submission.

    If True (default), the following metadata will be written to a JSON file:
    - When you submit the job, the job submission information (command, script path, number of jobs, config, and environment) will be written to a file named `submission.json` in the `meta` directory.
    - When the job starts running, the job ID and environment variables will be written to a file named `run.json` in the `meta` directory.
    """

    on_exit_script_support: bool
    """
    Whether to support running an on-exit script outside of srun.

    This is done by setting the environment variable `NSHRUNNER_EXIT_SCRIPT_DIR` to the path of an initially empty directory.
    Whenever the script wants something to be done on exit, it should write a bash script to this directory.
    """

    _exit_script_dir: Path
    """
    The directory to write the on-exit scripts to. (Internal use only)
    """


DEFAULT_KWARGS: SlurmJobKwargs = {
    "name": "nshrunner",
    "environment": {
        _env.SUBMIT_INTERFACE_MODULE: __name__,
    },
    # "nodes": 1,
    # "time": timedelta(hours=2),
    "timeout_signal": signal.SIGURG,
    "timeout_signal_delay": timedelta(seconds=90),
    "preempt_signal": signal.SIGTERM,
    "open_mode": "append",
    # "requeue": True,
    "emit_metadata": True,
    "on_exit_script_support": True,
}


def _determine_gres(kwargs: SlurmJobKwargs) -> Sequence[str] | None:
    """
    There are many different ways to specify GPU resources, but some are buggy.

    This function normalizes all other ways to specify GPU resources to the `gres` option.
    """

    # If `--gres` is set, just return it
    if (gres := kwargs.get("gres")) is not None:
        if isinstance(gres, str):
            gres = [gres]
        return gres

    # We will only support `--gpus` if `--nodes` is set to 1
    if (gpus := kwargs.get("gpus")) is not None:
        if kwargs.get("nodes") != 1:
            raise ValueError("Cannot specify `gpus` without `nodes` set to 1.")
        if isinstance(gpus, int):
            gpus = [f"gpu:{gpus}"]
        return gpus

    # `--gpus-per-task` is only supported if `--ntasks-per-node` is set (or can be inferred).
    if (gpus_per_task := kwargs.get("gpus_per_task")) is not None:
        if (ntasks_per_node := _determine_ntasks_per_node(kwargs)) is None:
            raise ValueError(
                "Cannot specify `gpus_per_task` without `ntasks_per_node`."
            )

        gpus_per_node = ntasks_per_node * gpus_per_task
        return [f"gpu:{gpus_per_node}"]

    # `--gpus-per-node` has no restrictions
    if (gpus_per_node := kwargs.get("gpus_per_node")) is not None:
        if isinstance(gpus_per_node, int):
            gpus_per_node = [f"gpu:{gpus_per_node}"]
        return gpus_per_node

    return None


def _determine_ntasks_per_node(kwargs: SlurmJobKwargs) -> int | None:
    """
    Determine the number of tasks per node, with validation.

    Raises
    ------
    RuntimeError
        If ntasks is specified without ntasks_per_node
    """
    # If ntasks is specified, raise an error suggesting ntasks_per_node
    if (ntasks := kwargs.get("ntasks")) is not None:
        raise RuntimeError(
            f"You set `--ntasks={ntasks}` in your SLURM bash script, but this variable "
            f"is not supported. HINT: Use `--ntasks-per-node={ntasks}` instead."
        )

    return kwargs.get("ntasks_per_node")


def _write_batch_script_to_file(
    path: Path,
    kwargs: SlurmJobKwargs,
    command: str,
    env: Mapping[str, str] = {},
    job_array_n_jobs: int | None = None,
):
    with path.open("w") as f:
        f.write("#!/bin/bash\n")

        if kwargs.get("requeue"):
            f.write("#SBATCH --requeue\n")

        if job_array_n_jobs is not None:
            f.write(f"#SBATCH --array=1-{job_array_n_jobs}\n")

        if (name := kwargs.get("name")) is not None:
            f.write(f"#SBATCH -J {name}\n")

        if (account := kwargs.get("account")) is not None:
            f.write(f"#SBATCH --account={account}\n")

        if (time := kwargs.get("time")) is not None:
            # A time limit of zero requests that no time limit be imposed. Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds".
            if time == 0:
                time_str = "0"
            else:
                total_seconds = time.total_seconds()
                hours, remainder = divmod(total_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                if hours > 24:
                    days, hours = divmod(hours, 24)
                    time_str = f"{int(days)}-{int(hours):02d}:{int(minutes):02d}"
                else:
                    time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
            f.write(f"#SBATCH --time={time_str}\n")

        if (nodes := kwargs.get("nodes")) is not None:
            f.write(f"#SBATCH --nodes={nodes}\n")

        if (ntasks_per_node := _determine_ntasks_per_node(kwargs)) is not None:
            f.write(f"#SBATCH --ntasks-per-node={ntasks_per_node}\n")

        if (output_file := kwargs.get("output_file")) is not None:
            output_file = str(Path(output_file).absolute())
            f.write(f"#SBATCH --output={output_file}\n")

        if (error_file := kwargs.get("error_file")) is not None:
            error_file = str(Path(error_file).absolute())
            f.write(f"#SBATCH --error={error_file}\n")

        if (partition := kwargs.get("partition")) is not None:
            if isinstance(partition, str):
                partition = [partition]
            f.write(f"#SBATCH --partition={','.join(partition)}\n")

        if (qos := kwargs.get("qos")) is not None:
            f.write(f"#SBATCH --qos={qos}\n")

        if (memory_mb := kwargs.get("memory_mb")) is not None:
            f.write(f"#SBATCH --mem={memory_mb}\n")

        if (memory_per_cpu_mb := kwargs.get("memory_per_cpu_mb")) is not None:
            f.write(f"#SBATCH --mem-per-cpu={memory_per_cpu_mb}\n")

        if (memory_per_gpu_mb := kwargs.get("memory_per_gpu_mb")) is not None:
            f.write(f"#SBATCH --mem-per-gpu={memory_per_gpu_mb}\n")

        if (cpus_per_task := kwargs.get("cpus_per_task")) is not None:
            f.write(f"#SBATCH --cpus-per-task={cpus_per_task}\n")

        if gres := _determine_gres(kwargs):
            f.write(f"#SBATCH --gres={','.join(gres)}\n")

        if (mail_user := kwargs.get("mail_user")) is not None:
            f.write(f"#SBATCH --mail-user={mail_user}\n")

        if (mail_type := kwargs.get("mail_type")) is not None:
            if isinstance(mail_type, str):
                mail_type = [mail_type]
            f.write(f"#SBATCH --mail-type={','.join(mail_type)}\n")

        if (dependency := kwargs.get("dependency")) is not None:
            f.write(f"#SBATCH --dependency={dependency}\n")

        if kwargs.get("exclusive"):
            f.write("#SBATCH --exclusive\n")

        if (open_mode := kwargs.get("open_mode")) is not None:
            f.write(f"#SBATCH --open-mode={open_mode}\n")

        if (constraint := kwargs.get("constraint")) is not None:
            if isinstance(constraint, str):
                constraint = [constraint]
            f.write(f"#SBATCH --constraint={','.join(constraint)}\n")

        if (signal := kwargs.get("timeout_signal")) is not None:
            signal_str = signal.name
            if (signal_delay := kwargs.get("timeout_signal_delay")) is not None:
                signal_str += f"@{math.ceil(signal_delay.total_seconds())}"
            f.write(f"#SBATCH --signal={signal_str}\n")

        f.write("\n")

        if env:
            for key, value in env.items():
                f.write(f"export {key}={value}\n")
            f.write("\n")

        if (
            setup_commands := kwargs.get("submission_script_setup_commands")
        ) is not None:
            for setup_command in setup_commands:
                f.write(f"{setup_command}\n")
            f.write("\n")

        if (command_prefix := kwargs.get("command_prefix")) is not None:
            command = " ".join(
                x_stripped
                for x in (command_prefix, command)
                if (x_stripped := x.strip())
            )
        if not kwargs.get("on_exit_script_support"):
            f.write(f"{command}\n")
        else:
            if (exit_script_dir := kwargs.get("_exit_script_dir")) is None:
                raise ValueError(
                    "on_exit_script_support is enabled, but _exit_script_dir is not set. "
                    "This is a logic error and should not happen."
                )

            f.write(f"{command} &\n")
            f.write("wait\n")

            # Add the on-exit script support
            # Basically, this just emits bash code that iterates
            # over all files in the exit script directory and runs them
            # in a subshell.
            emit_on_exit_commands(f, exit_script_dir)


def update_options(kwargs_in: SlurmJobKwargs, base_dir: Path):
    # Update the kwargs with the default values
    kwargs = copy.deepcopy(DEFAULT_KWARGS)

    # Merge the kwargs
    kwargs = cast(SlurmJobKwargs, always_merger.merge(kwargs, kwargs_in))
    del kwargs_in

    # If out/err files are not specified, set them
    logs_dir = base_dir / "logs"
    if kwargs.get("output_file") is None:
        logs_dir.mkdir(exist_ok=True, parents=True)
        kwargs["output_file"] = logs_dir / "output_%j_%a.out"
    if kwargs.get("error_file") is None:
        logs_dir.mkdir(exist_ok=True, parents=True)
        kwargs["error_file"] = logs_dir / "error_%j_%a.err"

    # Update the command_prefix to add srun:
    command_parts: list[str] = ["srun"]
    if (srun_flags := kwargs.get("srun_flags")) is not None:
        if isinstance(srun_flags, str):
            srun_flags = [srun_flags]
        command_parts.extend(srun_flags)

    # Add ntasks/cpus/gpus
    if (ntasks_per_node := _determine_ntasks_per_node(kwargs)) is not None:
        command_parts.append(f"--ntasks-per-node={ntasks_per_node}")

    if (cpus_per_task := kwargs.get("cpus_per_task")) is not None:
        command_parts.append(f"--cpus-per-task={cpus_per_task}")

    if gres := _determine_gres(kwargs):
        command_parts.append(f"--gres={','.join(gres)}")

    command_parts.append("--unbuffered")

    # Add the task id to the output filenames
    if (f := kwargs.get("output_file")) is not None:
        f = Path(f).absolute()
        command_parts.extend(
            [
                "--output",
                str(f.with_name(f"{f.stem}-%t{f.suffix}").absolute()),
            ]
        )
    if (f := kwargs.get("error_file")) is not None:
        f = Path(f).absolute()
        command_parts.extend(
            [
                "--error",
                str(f.with_name(f"{f.stem}-%t{f.suffix}").absolute()),
            ]
        )

    # If there is already a command prefix, combine them.
    if (existing_command_prefix := kwargs.get("command_prefix")) is not None:
        command_parts.extend(existing_command_prefix.split())
    # Add the command prefix to the kwargs.
    kwargs["command_prefix"] = " ".join(command_parts)

    # Set the default environment variables
    kwargs["environment"] = set_default_envs(
        kwargs.get("environment"),
        job_index="$SLURM_ARRAY_TASK_ID",
        local_rank="$SLURM_LOCALID",
        global_rank="$SLURM_PROCID",
        world_size="$SLURM_NTASKS",
        base_dir=base_dir,
        timeout_signal=kwargs.get("timeout_signal"),
        preempt_signal=kwargs.get("preempt_signal"),
    )

    # Emit the setup commands for run metadata
    if kwargs.get("emit_metadata"):
        kwargs["setup_commands"] = write_run_metadata_commands(
            kwargs.get("setup_commands"),
            is_worker_script=True,
        )
        kwargs["submission_script_setup_commands"] = write_run_metadata_commands(
            kwargs.get("submission_script_setup_commands"),
            is_worker_script=False,
        )

    # If `on_exit_script_support` is enabled, set the environment variable for EXIT_SCRIPT_DIR
    if kwargs.get("on_exit_script_support"):
        exit_script_dir = base_dir / "exit_scripts"
        exit_script_dir.mkdir(exist_ok=True)
        kwargs["environment"] = {
            **kwargs.get("environment", {}),
            _env.EXIT_SCRIPT_DIR: str(exit_script_dir.absolute()),
        }
        kwargs["_exit_script_dir"] = exit_script_dir

    return kwargs


def to_array_batch_script(
    command: str | Sequence[str],
    *,
    script_path: Path,
    num_jobs: int,
    config: SlurmJobKwargs,
    env: Mapping[str, str],
) -> Submission:
    """
    Create the batch script for the job.
    """
    if not isinstance(command, str):
        command = " ".join(command)

    # Write the submission information to a JSON file
    if config.get("emit_metadata"):
        write_submission_meta(
            script_path.parent,
            command=command,
            script_path=script_path,
            num_jobs=num_jobs,
            config=config,
            env=env,
        )

    # Write the batch script to the file
    _write_batch_script_to_file(
        script_path,
        config,
        command,
        env=env,
        job_array_n_jobs=num_jobs,
    )
    script_path = script_path.resolve().absolute()
    return Submission(
        command_parts=["sbatch", f"{script_path}"],
        script_path=script_path,
    )
