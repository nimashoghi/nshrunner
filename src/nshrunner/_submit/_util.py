import json
import signal
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
        return " ".join(self.command_parts)


def _write_submission_meta(
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
        )


def _write_run_meta(
    submit_dir: Path,
    *,
    job_id: int,
    env: Mapping[str, str],
):
    meta_dir = submit_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    with (meta_dir / RUN_META_FILE).open("w") as f:
        json.dump(
            {
                "job_id": job_id,
                "env": env,
            },
            f,
            indent=2,
        )


def _write_run_metadata_commands(
    setup_commands: Sequence[str] | None,
    is_worker_script: bool,
):
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

    if is_worker_script:
        meta_dir = f"${_env.SUBMIT_BASE_DIR}/meta/workers/${{{_env.GLOBAL_RANK}}}"
    else:
        meta_dir = f"${_env.SUBMIT_BASE_DIR}/meta"

    setup_commands.append(f'mkdir -p "{meta_dir}"')
    setup_commands.append(f'echo "${{{_env.JOB_INDEX}}}" > "{meta_dir}/job_id.txt"')
    setup_commands.append(f'env > "{meta_dir}/env.txt"')

    # Python-based JSON writing
    python_code = f"""
import json
import os

meta_dir = "{meta_dir}"
job_id = int(os.environ["{_env.JOB_INDEX}"])
env_vars = dict(os.environ)

with open(f"{{meta_dir}}/run.json", "w") as f:
    json.dump({{"job_id": job_id, "env": env_vars}}, f, indent=2)
""".strip()

    setup_commands.append(f'python3 -c "{python_code}"')

    return setup_commands


def _set_default_envs(
    env: Mapping[str, str] | None,
    *,
    job_index_env_var: str,
    local_rank_env_var: str | None,
    global_rank_env_var: str | None,
    world_size_env_var: str | None,
    base_dir: Path,
    timeout_signal: signal.Signals | None,
    preempt_signal: signal.Signals | None,
):
    env = dict(env) if env is not None else {}

    # Update the command to set JOB_INDEX to the job index variable (if exists)
    env = {_env.JOB_INDEX: f"${job_index_env_var}", **env}

    # Set the local rank, global rank, and world size environment variables
    if local_rank_env_var is not None:
        env = {_env.LOCAL_RANK: f"${local_rank_env_var}", **env}
    if global_rank_env_var is not None:
        env = {_env.GLOBAL_RANK: f"${global_rank_env_var}", **env}
    if world_size_env_var is not None:
        env = {_env.WORLD_SIZE: f"${world_size_env_var}", **env}

    # Add the current base directory to the environment variables
    env = {_env.SUBMIT_BASE_DIR: str(base_dir.resolve().absolute()), **env}

    # Update the environment variables to include the timeout signal
    if timeout_signal is not None:
        env = {_env.TIMEOUT_SIGNAL: timeout_signal.name, **env}

    # Update the environment variables to include the preempt signal
    if preempt_signal is not None:
        env = {_env.PREEMPT_SIGNAL: preempt_signal.name, **env}

    return env
