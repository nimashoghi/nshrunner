from __future__ import annotations

import logging
import os
import signal
from functools import cached_property
from pathlib import Path
from typing import Literal

from typing_extensions import assert_never, override

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


class Session:
    """Represents the current Runner execution session."""

    def __init__(self, session_id: str, session_dir: Path):
        """Initialize a Session with the required fields.

        Args:
            session_id: Unique identifier for the current session.
            session_dir: Directory path for the current session.
        """
        self.session_id = session_id
        """Unique identifier for the current session."""

        self.session_dir = session_dir
        """Directory path for the current session."""

    @override
    def __repr__(self) -> str:
        """Return a string representation of the Session.

        Only includes required fields and any cached properties that have been accessed.
        """
        # Always include required fields
        parts = [
            f"session_id={self.session_id!r}",
            f"session_dir={self.session_dir!r}",
        ]

        # Include any cached properties that have been accessed
        for attr_name in dir(self.__class__):
            if isinstance(getattr(self.__class__, attr_name, None), cached_property):
                # Check if this property has been accessed and cached
                if attr_name in self.__dict__:
                    value = self.__dict__[attr_name]
                    parts.append(f"{attr_name}={value!r}")

        return f"Session({', '.join(parts)})"

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

        return cls(
            session_id=session_id,
            session_dir=Path(session_dir),
        )

    @cached_property
    def snapshot_dir(self) -> Path | None:
        """Directory path for the snapshot, if available."""
        if snapshot_dir := os.environ.get(_env.SNAPSHOT_DIR):
            return Path(snapshot_dir)
        return None

    @cached_property
    def snapshot_modules(self) -> list[str] | None:
        """List of snapshot modules, if available."""
        if snapshot_modules := os.environ.get(_env.SNAPSHOT_MODULES):
            return snapshot_modules.split(",")
        return None

    @cached_property
    def is_worker_script(self) -> bool:
        """Indicates if this is running as a worker script."""
        return bool(int(os.environ.get(_env.IS_WORKER_SCRIPT, "0")))

    @cached_property
    def main_script_path(self) -> Path | None:
        """Path to the saved main script or notebook, if available."""
        if path := os.environ.get(_env.MAIN_SCRIPT_PATH):
            return Path(path)
        return None

    @cached_property
    def main_script_type(self) -> Literal["script", "notebook"] | None:
        """Type of the main script (Python script or Jupyter notebook), if available."""
        script_type_str = os.environ.get(_env.MAIN_SCRIPT_TYPE)
        if script_type_str == "script":
            return "script"
        elif script_type_str == "notebook":
            return "notebook"
        elif script_type_str is not None:
            log.warning(
                f"Unknown script type '{script_type_str}', expected 'script' or 'notebook'"
            )
        return None

    @cached_property
    def code_dir(self) -> Path | None:
        """Directory containing saved code and related artifacts."""
        if path := os.environ.get(_env.CODE_DIR):
            return Path(path)
        return None

    @cached_property
    def git_diff_path(self) -> Path | None:
        """Path to the saved git diff file, if available."""
        if path := os.environ.get(_env.GIT_DIFF_PATH):
            return Path(path)
        return None

    @cached_property
    def submit_base_dir(self) -> Path | None:
        """Base directory for job submission, if applicable."""
        if path := os.environ.get(_env.SUBMIT_BASE_DIR):
            return Path(path)
        return None

    @cached_property
    def submit_job_index(self) -> int | None:
        """Index of the current job in a job array, if applicable."""
        if job_index := os.environ.get(_env.SUBMIT_JOB_INDEX):
            return int(job_index)
        return None

    @cached_property
    def submit_timeout_signal(self) -> signal.Signals | None:
        """Signal number to be used for job timeout, if specified."""
        return _get_signal(_env.SUBMIT_TIMEOUT_SIGNAL)

    @cached_property
    def submit_preempt_signal(self) -> signal.Signals | None:
        """Signal number to be used for job preemption, if specified."""
        return _get_signal(_env.SUBMIT_PREEMPT_SIGNAL)

    @cached_property
    def submit_local_rank(self) -> int | None:
        """Local rank of the current process, if applicable."""
        if rank := os.environ.get(_env.SUBMIT_LOCAL_RANK):
            return int(rank)
        return None

    @cached_property
    def submit_global_rank(self) -> int | None:
        """Global rank of the current process, if applicable."""
        if rank := os.environ.get(_env.SUBMIT_GLOBAL_RANK):
            return int(rank)
        return None

    @cached_property
    def submit_world_size(self) -> int | None:
        """Total number of processes in the job, if applicable."""
        if size := os.environ.get(_env.SUBMIT_WORLD_SIZE):
            return int(size)
        return None

    @cached_property
    def exit_script_dir(self) -> Path | None:
        """Directory for exit scripts, if specified."""
        if path := os.environ.get(_env.EXIT_SCRIPT_DIR):
            return Path(path)
        return None

    @cached_property
    def submit_interface_module(self) -> str | None:
        """Name of the submit interface module, if specified."""
        return os.environ.get(_env.SUBMIT_INTERFACE_MODULE)

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
