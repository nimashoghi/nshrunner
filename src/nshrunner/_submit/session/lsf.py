import copy
import os
import signal
from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from logging import getLogger
from pathlib import Path
from typing import Any, Literal

from typing_extensions import TypeAlias, TypedDict, TypeVarTuple, Unpack

from ._output import SubmitOutput
from ._script import helper_script_to_command, write_helper_script

log = getLogger(__name__)


TArgs = TypeVarTuple("TArgs")

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

    command_template: str
    """
    The template for the command to execute the helper script.

    Default: `bash {/path/to/helper.sh}`.
    """

    signal: signal.Signals
    """
    The signal to send to the job as the "warning action".

    This corresponds to the "-wa" option in bsub.
    """

    signal_time: timedelta
    """
    The time (before the job ends) to send the signal.

    This corresponds to the "-wt" option in bsub.
    """

    # Our own custom options
    update_kwargs_fn: "Callable[[LSFJobKwargs], LSFJobKwargs]"
    """
    A function to update the kwargs with the defaults.

    This is useful for setting the command prefix to be dependent on num nodes/gpus/etc.
    """

    summit: bool
    """
    Whether the job is being submitted to Summit.

    If set to True, the job will be submitted to Summit and the default Summit options will be used.
    """

    load_job_step_viewer: bool
    """
    Whether to load the job step viewer.

    The job step viewer is a tool that can be used to view the job steps.
    """

    unset_cuda_visible_devices: bool
    """
    Whether to unset the CUDA_VISIBLE_DEVICES environment variable.

    This is a hack to fix issues with PyTorch Lightning and Summit.
    """

    unset_envs: Sequence[str]
    """
    A list of environment variables to unset.

    These environment variables will be unset before executing the job command.
    """

    force_envs: Mapping[str, str]
    """
    A dictionary of environment variables to force.

    These environment variables will be set before executing the job command, and any existing values will be overwritten.
    """


DEFAULT_KWARGS: LSFJobKwargs = {
    "name": "ll",
    # "nodes": 1,
    # "rs_per_node": 1,
    # "walltime": timedelta(hours=2),
    "summit": False,
    "signal": signal.SIGURG,
    "signal_time": timedelta(minutes=5),
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

    # Add ignoring of the CUDA_VISIBLE_DEVICES environment variable
    if kwargs.get("unset_cuda_visible_devices", False):
        # Regarding the --env_no_propagate=CUDA_VISIBLE_DEVICES flag:
        # PyTorch Lightning expects all GPUs to be present to all resource sets (tasks), but this is not the case
        #   when we use `jsrun -n6 -g1 -a1 -c7`. This is because `jsrun` automatically sets the `CUDA_VISIBLE_DEVICES`
        #   environment variable to the local rank of the task. PyTorch Lightning does not expect this and will fail
        #   with an error message like `RuntimeError: CUDA error: invalid device ordinal`. This hack will fix this by
        #   unsetting the `CUDA_VISIBLE_DEVICES` environment variable, so that PyTorch Lightning can see all GPUs.
        #   This is a hack and should be removed once PyTorch Lightning supports this natively.
        command_parts.append("--env_no_propagate=CUDA_VISIBLE_DEVICES")

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
    # "unset_cuda_visible_devices": True,
    "force_envs": {"JSM_NAMESPACE_LOCAL_RANK": "0"},
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

        if (signal := kwargs.get("signal")) is not None:
            # Convert the signal.Signals enum to a string
            signal = signal.name
            # Remove the "SIG" prefix
            signal = signal[len("SIG") :]
            f.write(f"#BSUB -wa {signal}\n")

        if (signal_time := kwargs.get("signal_time")) is not None:
            # Convert from time-delta to "H:M"
            total_seconds = signal_time.total_seconds()
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)

            signal_time = str(minutes)
            if hours > 0:
                signal_time = f"{hours}:{signal_time}"

            f.write(f"#BSUB -wt {signal_time}\n")

        f.write("\n")

        if kwargs.get("load_job_step_viewer", False):
            f.write("\n")
            f.write("module load job-step-viewer\n")

        f.write("\n")

        if (command_prefix := kwargs.get("command_prefix")) is not None:
            command = " ".join(
                x_stripped
                for x in (command_prefix, command)
                if (x_stripped := x.strip())
            )
        f.write(f"{command}\n")

    return path


def _update_kwargs(kwargs_in: LSFJobKwargs, base_dir: Path) -> LSFJobKwargs:
    # Update the kwargs with the default values
    kwargs = copy.deepcopy(DEFAULT_KWARGS)

    # If the job is being submitted to Summit, update the kwargs with the Summit defaults
    if kwargs_in.get("summit"):
        kwargs.update(SUMMIT_DEFAULTS)

    # Update the kwargs with the provided values
    kwargs.update(kwargs_in)
    del kwargs_in

    kwargs = _update_kwargs_jsrun(kwargs, base_dir)

    if (update_kwargs_fn := kwargs.get("update_kwargs_fn")) is not None:
        kwargs = copy.deepcopy(update_kwargs_fn(kwargs))

    return kwargs


def to_array_batch_script(
    dest: Path,
    callable: Callable[[Unpack[TArgs]], Any],
    args_list: Sequence[tuple[Unpack[TArgs]]],
    /,
    job_index_variable: str = "LSB_JOBINDEX",
    print_environment_info: bool = False,
    python_command_prefix: str | None = None,
    **kwargs: Unpack[LSFJobKwargs],
) -> SubmitOutput:
    """
    Create the batch script for the job.
    """

    from ...picklerunner import serialize_many

    kwargs = _update_kwargs(kwargs, dest)

    # Convert the command/callable to a string for the command
    num_jobs = len(args_list)

    destdir = dest / "fns"
    destdir.mkdir(exist_ok=True)

    additional_command_parts: list[str] = []
    if kwargs.get("unset_cuda_visible_devices", False):
        additional_command_parts.append("--unset-cuda")

    if (unset_envs := kwargs.get("unset_envs")) is not None:
        for env in unset_envs:
            additional_command_parts.append(f"--unset-env {env}")

    if (force_envs := kwargs.get("force_envs")) is not None:
        for key, value in force_envs.items():
            additional_command_parts.append(f"--force-env {key}={value}")

    serialized_command = serialize_many(
        destdir,
        callable,
        [(args, {}) for args in args_list],
        start_idx=1,  # LSF job indices are 1-based
        additional_command_parts=additional_command_parts,
    )
    helper_path = write_helper_script(
        destdir,
        serialized_command.to_bash_command(
            job_index_variable, print_environment_info=print_environment_info
        ),
        kwargs.get("environment", {}),
        kwargs.get("setup_commands", []),
        command_prefix=python_command_prefix,
    )
    command = helper_script_to_command(helper_path, kwargs.get("command_template"))

    script_path = _write_batch_script_to_file(
        dest / "launch.sh",
        kwargs,
        command,
        job_array_n_jobs=num_jobs,
    )
    script_path = script_path.resolve().absolute()
    return SubmitOutput(
        command_parts=["bsub", str(script_path)],
        script_path=script_path,
    )
