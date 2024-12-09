from __future__ import annotations

import copy
import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

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


class ScreenJobKwargs(TypedDict, total=False):
    name: str
    """
    The name of the job.
    """

    screen_log_file: _Path
    """
    The path to the log file to write the output to. This is done by setting the `-L` flag to `screen`.
    """

    output_file: _Path
    """
    The file to write the job output to.
    """

    error_file: _Path
    """
    The file to write the job errors to.
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

    attach: bool
    """
    Whether to attach to the screen session after starting it.
    """

    pause_before_exit: bool
    """
    Whether to pause before exiting the screen session.
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


DEFAULT_KWARGS: ScreenJobKwargs = {
    "name": "nshrunner",
    "environment": {
        _env.SUBMIT_INTERFACE_MODULE: __name__,
    },
    "attach": True,
    "pause_before_exit": True,
    "emit_metadata": True,
    "on_exit_script_support": True,
}


def _write_batch_script_to_file(
    path: Path,
    kwargs: ScreenJobKwargs,
    command: str,
    env: Mapping[str, str] = {},
):
    with path.open("w") as f:
        f.write("#!/bin/bash\n")

        f.write("\n")

        # Set up logging
        output_file = kwargs.get("output_file")
        error_file = kwargs.get("error_file")
        if output_file is not None and error_file is not None:
            # Ex: exec > >(tee -a stdout.log) 2> >(tee -a stderr.log >&2)
            f.write(f"exec > >(tee -a {output_file}) 2> >(tee -a {error_file} >&2)\n")
        elif output_file is not None and error_file is None:
            f.write(f"exec > >(tee -a {output_file})\n")
        elif output_file is None and error_file is not None:
            f.write(f"exec 2> >(tee -a {error_file} >&2)\n")

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


def update_options(kwargs_in: ScreenJobKwargs, base_dir: Path):
    # Update the kwargs with the default values
    kwargs = copy.deepcopy(DEFAULT_KWARGS)

    # Merge the kwargs
    kwargs = cast(ScreenJobKwargs, always_merger.merge(kwargs, kwargs_in))
    del kwargs_in

    # If out/err files are not specified, set them
    logs_dir = base_dir / "logs"
    if kwargs.get("screen_log_file") is None:
        logs_dir.mkdir(exist_ok=True, parents=True)
        kwargs["screen_log_file"] = logs_dir / "session.log"
    if kwargs.get("output_file") is None:
        logs_dir.mkdir(exist_ok=True, parents=True)
        kwargs["output_file"] = base_dir / "output.log"
    if kwargs.get("error_file") is None:
        logs_dir.mkdir(exist_ok=True, parents=True)
        kwargs["error_file"] = base_dir / "error.log"

    # Set the default environment variables
    kwargs["environment"] = set_default_envs(
        kwargs.get("environment"),
        job_index=None,
        local_rank="0",
        global_rank="0",
        world_size="1",
        base_dir=base_dir,
        timeout_signal=None,
        preempt_signal=None,
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


def _launch_session(
    session_command: list[str],
    *,
    log_file: Path | None,
    session_name: str,
    attach: bool,
):
    logging_args: list[str] = []
    if log_file is not None:
        logging_args = [
            "-L",
            "-Logfile",
            str(log_file.absolute()),
        ]
    return [
        "screen",
        "-dmS" if not attach else "-S",
        session_name,
        # Save the logs to a file
        *logging_args,
        # Enable UTF-8 encoding
        "-U",
        *session_command,
    ]


def to_array_batch_script(
    command: str | Sequence[str],
    *,
    script_path: Path,
    config: ScreenJobKwargs,
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
            num_jobs=1,
            config=config,
            env=env,
        )

    # Write the batch script to the file
    _write_batch_script_to_file(
        script_path,
        config,
        command,
        env=env,
    )
    script_path.chmod(0o755)

    return Submission(
        command_parts=_launch_session(
            session_command=["bash", str(script_path.resolve().absolute())],
            log_file=Path(log_file)
            if (log_file := config.get("screen_log_file")) is not None
            else None,
            session_name=config.get("name", "nshrunner"),
            attach=config.get("attach", True),
        ),
        script_path=script_path,
    )
