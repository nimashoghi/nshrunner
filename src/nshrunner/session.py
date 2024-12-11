from __future__ import annotations

import logging
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from typing_extensions import assert_never

from . import _env

log = logging.getLogger(__name__)


def _get_signal(env_var: str):
    if not (sig_name := os.environ.get(env_var)):
        return None

    try:
        return signal.Signals[sig_name]
    except KeyError:
        log.warning(
            f"Invalid signal name '{sig_name}' in environment variable '{env_var}'"
        )
        return None


@dataclass
class Session:
    """Represents the current Runner execution session."""

    session_id: str
    """Unique identifier for the current session."""

    session_dir: Path
    """Directory path for the current session."""

    snapshot_dir: Path | None
    """Directory path for the snapshot, if available."""

    snapshot_modules: list[str] | None
    """List of snapshot modules, if available."""

    is_worker_script: bool
    """Indicates if this is running as a worker script."""

    submit_base_dir: Path | None
    """Base directory for job submission, if applicable."""

    submit_job_index: int | None
    """Index of the current job in a job array, if applicable."""

    submit_timeout_signal: signal.Signals | None
    """Signal number to be used for job timeout, if specified."""

    submit_preempt_signal: signal.Signals | None
    """Signal number to be used for job preemption, if specified."""

    submit_local_rank: int | None
    """Local rank of the current process, if applicable."""

    submit_global_rank: int | None
    """Global rank of the current process, if applicable."""

    submit_world_size: int | None
    """Total number of processes in the job, if applicable."""

    exit_script_dir: Path | None
    """Directory for exit scripts, if specified."""

    submit_interface_module: str | None
    """Name of the submit interface module, if specified."""

    @classmethod
    def from_current_session(cls):
        """
        Create a Session instance from the current environment variables.
        Returns None if not in a Runner session.
        """
        if not (session_id := os.environ.get(_env.SESSION_ID)):
            return None
        if not (session_dir := os.environ.get(_env.SESSION_DIR)):
            return None

        snapshot_dir = os.environ.get(_env.SNAPSHOT_DIR)
        snapshot_modules = os.environ.get(_env.SNAPSHOT_MODULES)

        return cls(
            session_id=session_id,
            session_dir=Path(session_dir),
            snapshot_dir=Path(snapshot_dir) if snapshot_dir else None,
            snapshot_modules=snapshot_modules.split(",") if snapshot_modules else None,
            is_worker_script=bool(int(os.environ.get(_env.IS_WORKER_SCRIPT, "0"))),
            submit_base_dir=Path(p)
            if (p := os.environ.get(_env.SUBMIT_BASE_DIR))
            else None,
            submit_job_index=int(job_index)
            if (job_index := os.environ.get(_env.SUBMIT_JOB_INDEX))
            else None,
            submit_timeout_signal=_get_signal(_env.SUBMIT_TIMEOUT_SIGNAL),
            submit_preempt_signal=_get_signal(_env.SUBMIT_PREEMPT_SIGNAL),
            submit_local_rank=int(rank)
            if (rank := os.environ.get(_env.SUBMIT_LOCAL_RANK))
            else None,
            submit_global_rank=int(rank)
            if (rank := os.environ.get(_env.SUBMIT_GLOBAL_RANK))
            else None,
            submit_world_size=int(size)
            if (size := os.environ.get(_env.SUBMIT_WORLD_SIZE))
            else None,
            exit_script_dir=Path(p)
            if (p := os.environ.get(_env.EXIT_SCRIPT_DIR))
            else None,
            submit_interface_module=os.environ.get(_env.SUBMIT_INTERFACE_MODULE),
        )

    def write_exit_script(
        self,
        script_name: str,
        script_contents: str,
        on_error: Literal["warn", "raise"] = "warn",
    ):
        """
        Write an exit script to the session directory.
        """
        if not self.exit_script_dir:
            error_msg = "No exit script directory specified"
            match on_error:
                case "raise":
                    raise ValueError(error_msg)
                case "warn":
                    log.warning(error_msg)
                case _:
                    assert_never(on_error)
            return

        script_path = self.exit_script_dir / script_name
        with open(script_path, "w") as f:
            f.write(script_contents)
        os.chmod(script_path, 0o755)
        log.info(f"Wrote exit script to {script_path}")
