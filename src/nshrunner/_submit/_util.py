import json
from collections.abc import Mapping
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


def _emit_write_run_metadata_commands(
    *,
    submit_dir_env_var: str = _env.SUBMIT_BASE_DIR,
    job_id_env_var: str = _env.JOB_INDEX,
) -> list[str]:
    """
    Creates a list of bash commands that will write the run metadata
    to the submission directory. Specifically:
    - We want to save the job id (in plain text) to a file called `job_id.txt`
    - We want to save the environment variables to a file called `env.txt`
    - If Python exists, we'll also create a run.json file
    """
    setup_commands = []

    setup_commands.append(f'mkdir -p "${submit_dir_env_var}/meta"')
    setup_commands.append(
        f'echo "${job_id_env_var}" > "${submit_dir_env_var}/meta/job_id.txt"'
    )
    setup_commands.append(f'env > "${submit_dir_env_var}/meta/env.txt"')

    # Add Python-based JSON writing if Python exists
    setup_commands.append(
        f"""
if command -v python >/dev/null 2>&1; then
    python -c "import json, os; json.dump({'job_id': int(os.environ['{job_id_env_var}']), 'env': dict(os.environ)}, open(os.environ['{submit_dir_env_var}']+'/meta/run.json', 'w'), indent=2)"
fi
""".strip()
    )

    return setup_commands
