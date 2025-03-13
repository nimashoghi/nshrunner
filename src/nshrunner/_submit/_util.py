from __future__ import annotations

import json
import signal
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from typing import Any

from .. import _env

SUBMISSION_META_FILE = "submission.json"
RUN_META_FILE = "run.json"


@dataclass(frozen=True)
class Submission:
    command_parts: list[str]
    script_path: Path

    @property
    def command_str(self) -> str:
        """
        Convert command parts to a string, adding quotes around arguments with spaces.

        Returns
        -------
        str
            Command string with proper quoting for arguments containing spaces or other whitespace
        """
        quoted_parts = []
        for part in self.command_parts:
            # Check if the part contains any whitespace characters
            if any(char.isspace() for char in part):
                quoted_parts.append(f'"{part}"')
            else:
                quoted_parts.append(part)
        return " ".join(quoted_parts)


def write_submission_meta(
    submit_dir: Path,
    *,
    command: str,
    script_path: Path,
    num_jobs: int,
    config: Mapping[str, Any],
    env: Mapping[str, str],
):
    meta_dir = submit_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    with (meta_dir / SUBMISSION_META_FILE).open("w") as f:
        json.dump(
            {
                "command": command,
                "script_path": str(script_path),
                "num_jobs": num_jobs,
                "config": config,
                "env": env,
            },
            f,
            indent=2,
            default=str,
        )


def set_default_envs(
    existing_env: Mapping[str, str] | None,
    /,
    *,
    job_index: str | None,
    local_rank: str,
    global_rank: str,
    world_size: str,
    base_dir: Path,
    timeout_signal: signal.Signals | None,
    preempt_signal: signal.Signals | None,
):
    env: dict[str, str] = {}

    # Update the command to set JOB_INDEX to the job index variable (if exists)
    env[_env.SUBMIT_JOB_INDEX] = str(job_index) if job_index is not None else ""

    # Set the local rank, global rank, and world size environment variables
    env[_env.SUBMIT_LOCAL_RANK] = str(local_rank)
    env[_env.SUBMIT_GLOBAL_RANK] = str(global_rank)
    env[_env.SUBMIT_WORLD_SIZE] = str(world_size)

    # Add the current base directory to the environment variables
    env[_env.SUBMIT_BASE_DIR] = str(base_dir.resolve().absolute())

    # Update the environment variables to include the timeout signal
    env[_env.SUBMIT_TIMEOUT_SIGNAL] = (
        timeout_signal.name if timeout_signal is not None else ""
    )

    # Update the environment variables to include the preempt signal
    env[_env.SUBMIT_PREEMPT_SIGNAL] = (
        preempt_signal.name if preempt_signal is not None else ""
    )

    return {**env, **(existing_env or {})}


def write_run_metadata_commands(
    setup_commands: Sequence[str] | None,
    *,
    is_worker_script: bool = False,
) -> list[str]:
    """
    Creates a list of bash commands that will write the run metadata
    to the submission directory. For the parent script:
    - We want to save the job id (in plain text) to a file called `job_id.txt`
    - We want to save the environment variables to a file called `env.txt`
    - We'll also create a run.json file
    For worker scripts:
    - We create the same files, but in a /meta/workers/{rank}/ directory
    """
    setup_commands = list(setup_commands) if setup_commands is not None else []

    setup_commands.append("")
    setup_commands.append("")
    comment = "# Run metadata"
    comment += " (worker)" if is_worker_script else " (parent)"
    setup_commands.append(comment)

    if is_worker_script:
        meta_dir = (
            f"${_env.SUBMIT_BASE_DIR}/meta/workers/${{{_env.SUBMIT_GLOBAL_RANK}}}"
        )
    else:
        meta_dir = f"${_env.SUBMIT_BASE_DIR}/meta/parent"

    setup_commands.append(f'mkdir -p "{meta_dir}"')
    setup_commands.append(
        f'echo "${{{_env.SUBMIT_JOB_INDEX}}}" > "{meta_dir}/job_id.txt"'
    )
    setup_commands.append(f'env > "{meta_dir}/env.txt"')

    # Python-based JSON writing
    python_code = f"""
import json
import os

meta_dir = os.environ['{_env.SUBMIT_BASE_DIR}']
if {is_worker_script}:
    meta_dir = os.path.join(meta_dir, 'meta', 'workers', os.environ['{_env.SUBMIT_GLOBAL_RANK}'])
else:
    meta_dir = os.path.join(meta_dir, 'meta')

job_id = os.environ['{_env.SUBMIT_JOB_INDEX}']
env_vars = dict(os.environ)

with open(os.path.join(meta_dir, 'run.json'), 'w') as f:
    json.dump({{'job_id': job_id, 'env': env_vars}}, f, indent=2)
""".strip()

    setup_commands.append(f'python -c "{python_code}"')

    setup_commands.append(comment.replace("# R", "# End r", 1))
    setup_commands.append("")
    setup_commands.append("")

    return setup_commands


ON_EXIT_TEMPLATE = r"""
# Execute on-exit scripts
exit_script_dir="{exit_script_dir}"
exit_scripts=("$exit_script_dir"/*)
num_scripts=${{#exit_scripts[@]}}
echo "Found $num_scripts on-exit script(s) in $exit_script_dir"
for script in "${{exit_scripts[@]}}"; do
    if [ -f "$script" ]; then
        echo "Executing on-exit script: $script"
        if [ -x "$script" ]; then
            "$script"
        else
            bash "$script"
        fi
    fi
done""".strip()


def emit_on_exit_commands(f: TextIOWrapper, exit_script_dir: Path):
    # Add the on-exit script support
    # Basically, this just emits bash code that iterates
    # over all files in the exit script directory and runs them
    # in a subshell.
    f.write(
        ON_EXIT_TEMPLATE.format(
            exit_script_dir=str(exit_script_dir.resolve().absolute())
        )
        + "\n"
    )
