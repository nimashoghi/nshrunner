from __future__ import annotations

import copy
import logging
import os
import signal
from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Literal, cast

from deepmerge import always_merger
from typing_extensions import TypeAlias, TypedDict

from .. import _env
from ._util import (
    Submission,
    _emit_on_exit_commands,
    _set_default_envs,
    _write_run_metadata_commands,
    _write_submission_meta,
)

log = logging.getLogger(__name__)


_Path: TypeAlias = str | Path | os.PathLike


class LSFJobKwargs(TypedDict, total=False):
    name: str
    """
    The name of the job.

    This corresponds to the "-J" option in bsub.
    """

    queue: str | Sequence[str]
    """
    The queue to submit the job to.

    This corresponds to the "-q" option in bsub. If not specified, the default queue will be used.
    """

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

    walltime: timedelta
    """
    The maximum walltime for the job.

    This corresponds to the "-W" option in bsub. The format is "HH:MM" or "MM". If not specified, the default walltime will be used.
    """

    memory_mb: int
    """
    The maximum memory for the job in MB.

    This corresponds to the "-M" option in bsub. If not specified, the default memory limit will be used.
    """

    cpu_limit: int
    """
    The CPU time limit for the job in minutes.

    This corresponds to the "-c" option in bsub. If not specified, the default CPU limit will be used.
    """

    rerunnable: bool
    """
    Whether the job should be rerunnable.

    This corresponds to the "-r" option in bsub. If set to True, the job will be rerun if it fails due to a system failure.
    """

    dependency_conditions: Sequence[str]
    """
    The dependency conditions for the job.

    This corresponds to the "-w" option in bsub. Each condition is a string that specifies the dependency condition.
    Multiple conditions can be specified, and they will be combined using logical AND.
    """

    email: str
    """
    The email address to send notifications to.

    This corresponds to the "-u" option in bsub. If specified, job notifications will be sent to this email address.
    """

    notify_begin: bool
    """
    Whether to send a notification when the job begins.

    This corresponds to the "-B" option in bsub. If set to True, a notification will be sent when the job begins.
    """

    notify_end: bool
    """
    Whether to send a notification when the job ends.

    This corresponds to the "-N" option in bsub. If set to True, a notification will be sent when the job ends.
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

    project: str
    """
    The project to charge the job to.

    This corresponds to the "-P" option in bsub. If specified, the job will be charged to this project.
    """

    nodes: int
    """
    The number of nodes to use for the job.

    This corresponds to the "-nnodes" option in bsub. The default is 1 node.
    """

    rs_per_node: int
    """
    The number of resource sets per node. This is sent to the `jsrun` command.
    """

    cpus_per_rs: int | Literal["ALL_CPUS"]
    """
    The number of CPUs per resource set. This is sent to the `jsrun` command.
    """

    gpus_per_rs: int | Literal["ALL_GPUS"]
    """
    The number of GPUs per resource set. This is sent to the `jsrun` command.
    """

    tasks_per_rs: int
    """
    The number of tasks per resource set. This is sent to the `jsrun` command.
    """

    alloc_flags: str
    """
    The allocation flags for the job.

    This corresponds to the "-alloc_flags" option in bsub. If specified, the job will be allocated using these flags.
    """

    command_prefix: str
    """
    A command to prefix the job command with.

    This is used to add commands like `jsrun` to the job command.
    """

    timeout_signal: signal.Signals
    """
    The signal to send to the job as the "warning action".

    This corresponds to the "-wa" option in bsub.
    """

    preempt_signal: signal.Signals
    """
    The signal to send to the job when it is preempted.
    """

    timeout_signal_time: timedelta
    """
    The time (before the job ends) to send the signal.

    This corresponds to the "-wt" option in bsub.
    """

    login_shell: str
    """
    The login shell to use for the job.

    This corresponds to the "-L" option in bsub. If specified, the job will be run using this login shell.
    """

    on_exit_script_support: bool
    """
    Whether to support running an on-exit script outside of jsrun.

    This is done by setting the environment variable `NSHRUNNER_EXIT_SCRIPT_DIR` to the path of an initially empty directory.
    Whenever the script wants something to be done on exit, it should write a bash script to this directory.
    """

    emit_metadata: bool
    """
    Whether to emit metadata about the job submission.

    If True (default), the following metadata will be written to a JSON file:
    - When you submit the job, the job submission information (command, script path, number of jobs, config, and environment) will be written to a file named `submission.json` in the `meta` directory.
    - When the job starts running, the job ID and environment variables will be written to a file named `run.json` in the `meta` directory.
    """

    # Our own custom options
    summit: bool
    """
    Whether the job is being submitted to Summit.

    If set to True, the job will be submitted to Summit and the default Summit options will be used.
    """

    _exit_script_dir: Path
    """
    The directory to write the on-exit scripts to. (Internal use only)
    """


DEFAULT_KWARGS: LSFJobKwargs = {
    "name": "nshrunner",
    "environment": {
        _env.SUBMIT_INTERFACE_MODULE: __name__,
    },
    # "nodes": 1,
    # "rs_per_node": 1,
    # "walltime": timedelta(hours=2),
    "summit": False,
    # On SIGURG:
    # Important note from https://amrex-astro.github.io/workflow/olcf-workflow.html:
    # We can also ask the job manager to send a warning signal some amount of time before the allocation expires by passing -wa 'signal' and -wt '[hour:]minute' to bsub. We can then have bash create a dump_and_stop file when it receives the signal, which will tell Castro to output a checkpoint file and exit cleanly after it finishes the current timestep. An important detail that I couldn't find documented anywhere is that the job manager sends the signal to all the processes in the job, not just the submission script, and we have to use a signal that is ignored by default so Castro doesn't immediately crash upon receiving it. SIGCHLD, SIGURG, and SIGWINCH are the only signals that fit this requirement and of these, SIGURG is the least likely to be triggered by other events.
    "timeout_signal": signal.SIGURG,
    "timeout_signal_time": timedelta(minutes=5),
    "on_exit_script_support": True,
    "emit_metadata": True,
}


def _update_kwargs_jsrun(kwargs: LSFJobKwargs, base_dir: Path) -> LSFJobKwargs:
    kwargs = copy.deepcopy(kwargs)

    # Update the command_prefix to add srun:
    command_parts: list[str] = ["jsrun"]

    # Add the worker logs
    command_parts.extend(["--stdio_mode", "individual"])
    command_parts.extend(
        ["--stdio_stdout", str(base_dir / "logs" / "worker_out.%h.%j.%t.%p")]
    )
    command_parts.extend(
        ["--stdio_stderr", str(base_dir / "logs" / "worker_err.%h.%j.%t.%p")]
    )

    if (rs_per_node := kwargs.get("rs_per_node")) is not None:
        # Add the total number of resource sets requested across all nodes in the job
        total_num_rs = rs_per_node * kwargs.get("nodes", 1)
        command_parts.append(f"-n{total_num_rs}")

        # Add the number of resource sets requested on each node
        command_parts.append(f"-r{rs_per_node}")

    # Add the number of CPUs per resource set
    if (cpus_per_rs := kwargs.get("cpus_per_rs")) is not None:
        command_parts.append(f"-c{cpus_per_rs}")

    # Add the number of GPUs per resource set
    if (gpus_per_rs := kwargs.get("gpus_per_rs")) is not None:
        command_parts.append(f"-g{gpus_per_rs}")

    # Add the number of tasks per resource set
    if (tasks_per_rs := kwargs.get("tasks_per_rs")) is not None:
        command_parts.append(f"-a{tasks_per_rs}")

    # Add -brs. This binds the resource sets to the CPUs.
    command_parts.append("-brs")

    # If there is already a command prefix, combine them.
    if (existing_command_prefix := kwargs.get("command_prefix")) is not None:
        command_parts.extend(existing_command_prefix.split())

    # Add the command prefix to the kwargs.
    kwargs["command_prefix"] = " ".join(command_parts)

    return kwargs


SUMMIT_DEFAULTS: LSFJobKwargs = {
    "environment": {"JSM_NAMESPACE_LOCAL_RANK": "0"},
    "rs_per_node": 6,
    "cpus_per_rs": 7,
    "gpus_per_rs": 1,
    "tasks_per_rs": 1,
}


def _append_job_index_to_path(path: Path) -> Path:
    # If job array, append the job index to the output file
    # E.g., if `output_file` is "output_%J.out", we want "output_%J_%I.out"
    stem = path.stem
    suffix = path.suffix
    new_stem = f"{stem}_%I"
    new_path = path.with_name(new_stem + suffix)
    return new_path


def _write_batch_script_to_file(
    path: Path,
    kwargs: LSFJobKwargs,
    command: str,
    env: Mapping[str, str] = {},
    job_array_n_jobs: int | None = None,
):
    logs_base = path.parent / "logs"
    logs_base.mkdir(exist_ok=True)

    if kwargs.get("output_file") is None:
        kwargs["output_file"] = logs_base / "output_%J_%I.out"

    if kwargs.get("error_file") is None:
        kwargs["error_file"] = logs_base / "error_%J_%I.err"

    with path.open("w") as f:
        f.write("#!/bin/bash\n")

        if (login_shell := kwargs.get("login_shell")) is not None:
            f.write(f"#BSUB -L {login_shell}\n")

        if (name := kwargs.get("name")) is not None:
            if job_array_n_jobs is not None:
                name += "[1-" + str(job_array_n_jobs) + "]"
            f.write(f"#BSUB -J {name}\n")

        if (project := kwargs.get("project")) is not None:
            f.write(f"#BSUB -P {project}\n")

        if (walltime := kwargs.get("walltime")) is not None:
            # Convert the walltime to the format expected by LSF:
            # -W [hour:]minute[/host_name | /host_model]
            # E.g., 72 hours -> 72:00
            total_minutes = walltime.total_seconds() // 60
            hours = int(total_minutes // 60)
            minutes = int(total_minutes % 60)
            walltime = f"{hours:02d}:{minutes:02d}"
            f.write(f"#BSUB -W {walltime}\n")

        if (nodes := kwargs.get("nodes")) is not None:
            f.write(f"#BSUB -nnodes {nodes}\n")

        if (output_file := kwargs.get("output_file")) is not None:
            output_file = Path(output_file).absolute()
            if job_array_n_jobs is not None:
                output_file = _append_job_index_to_path(output_file)
            output_file = str(output_file)
            f.write(f"#BSUB -o {output_file}\n")

        if (error_file := kwargs.get("error_file")) is not None:
            error_file = Path(error_file).absolute()
            if job_array_n_jobs is not None:
                error_file = _append_job_index_to_path(error_file)
            error_file = str(error_file)
            f.write(f"#BSUB -e {error_file}\n")

        if (queue := kwargs.get("queue")) is not None:
            if not isinstance(queue, str) and isinstance(queue, Sequence):
                assert len(queue) == 1, "Only one queue can be specified"
                queue = queue[0]
            f.write(f"#BSUB -q {queue}\n")

        if (memory_mb := kwargs.get("memory_mb")) is not None:
            f.write(f"#BSUB -M {memory_mb}\n")

        if (cpu_limit := kwargs.get("cpu_limit")) is not None:
            f.write(f"#BSUB -c {cpu_limit}\n")

        if (rerunnable := kwargs.get("rerunnable")) is not None:
            f.write(f"#BSUB -r {'y' if rerunnable else 'n'}\n")

        for dependency_condition in kwargs.get("dependency_conditions", []):
            f.write(f"#BSUB -w {dependency_condition}\n")

        if (email := kwargs.get("email")) is not None:
            f.write(f"#BSUB -u {email}\n")

        if (notify_begin := kwargs.get("notify_begin")) is not None:
            f.write(f"#BSUB -B {'y' if notify_begin else 'n'}\n")

        if (notify_end := kwargs.get("notify_end")) is not None:
            f.write(f"#BSUB -N {'y' if notify_end else 'n'}\n")

        if (alloc_flags := kwargs.get("alloc_flags")) is not None:
            f.write(f"#BSUB -alloc_flags {alloc_flags}\n")

        if (signal := kwargs.get("timeout_signal")) is not None:
            # Convert the signal.Signals enum to a string
            signal = signal.name
            # Remove the "SIG" prefix
            signal = signal[len("SIG") :]
            f.write(f"#BSUB -wa {signal}\n")

        if (signal_time := kwargs.get("timeout_signal_time")) is not None:
            # Convert from time-delta to "H:M"
            total_seconds = signal_time.total_seconds()
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)

            signal_time = str(minutes)
            if hours > 0:
                signal_time = f"{hours}:{signal_time}"

            f.write(f"#BSUB -wt {signal_time}\n")

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
            f.write("jswait all\n")

            # Add the on-exit script support
            # Basically, this just emits bash code that iterates
            # over all files in the exit script directory and runs them
            # in a subshell.
            _emit_on_exit_commands(f, exit_script_dir)

    return path


def update_options(kwargs_in: LSFJobKwargs, base_dir: Path) -> LSFJobKwargs:
    # Update the kwargs with the default values
    global DEFAULT_KWARGS
    kwargs = copy.deepcopy(DEFAULT_KWARGS)

    # If the job is being submitted to Summit, update the kwargs with the Summit defaults
    if kwargs_in.get("summit"):
        global SUMMIT_DEFAULTS
        kwargs = cast(LSFJobKwargs, always_merger.merge(kwargs, SUMMIT_DEFAULTS))

    # Update the kwargs with the provided values
    kwargs = cast(LSFJobKwargs, always_merger.merge(kwargs, kwargs_in))
    del kwargs_in

    # Update the kwargs to set the command prefix for jsrun
    kwargs = _update_kwargs_jsrun(kwargs, base_dir)

    # Set the default environment variables
    kwargs["environment"] = _set_default_envs(
        kwargs.get("environment"),
        job_index="$LSB_JOBINDEX",
        local_rank="$JSM_NAMESPACE_LOCAL_RANK",
        global_rank="$JSM_NAMESPACE_GLOBAL_RANK",
        world_size="$JSM_NAMESPACE_GLOBAL_SIZE",
        base_dir=base_dir,
        timeout_signal=kwargs.get("timeout_signal"),
        preempt_signal=kwargs.get("preempt_signal"),
    )

    # Emit the setup commands for run metadata
    if kwargs.get("emit_metadata"):
        kwargs["setup_commands"] = _write_run_metadata_commands(
            kwargs.get("setup_commands"),
            is_worker_script=True,
        )
        kwargs["submission_script_setup_commands"] = _write_run_metadata_commands(
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
    config: LSFJobKwargs,
    env: Mapping[str, str],
) -> Submission:
    """
    Create the batch script for the job.
    """
    if not isinstance(command, str):
        command = " ".join(command)

    # Write the submission information to a JSON file
    if config.get("emit_metadata"):
        _write_submission_meta(
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
        command_parts=["bsub", str(script_path)],
        script_path=script_path,
    )
