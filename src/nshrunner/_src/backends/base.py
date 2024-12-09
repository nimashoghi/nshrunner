from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import nshconfig as C
from strenum import StrEnum

from ..picklerunner import SerializedArgs, SerializedCallable


@dataclass(frozen=True)
class JobInfo:
    id: str
    """The job ID."""


class JobStatus(StrEnum):
    COMPLETED = "COMPLETED"
    """The job has completed successfully."""

    RUNNING = "RUNNING"
    """The job is currently running."""

    FAILED = "FAILED"
    """The job has failed."""


@dataclass(frozen=True)
class JobStatusInfo:
    """Represents the status of a job."""

    status: JobStatus
    """The job status."""

    description: str | None = None
    """Any additional information about the job status."""


@runtime_checkable
class Job(Protocol):
    @property
    def info(self) -> JobInfo:
        """Get information about the job.

        Returns
        -------
        JobInfo
            The job information
        """
        ...

    def status(self) -> JobStatusInfo:
        """Get the status of the job.

        Returns
        -------
        JobStatus
            The job status
        """
        ...

    def results(self) -> list[Any]:
        """Get the results of the job.

        Returns
        -------
        list[Any]
            The results of the job
        """
        ...


class BaseBackendConfig(C.Config, ABC):
    @abstractmethod
    def create_backend(self) -> BaseBackend:
        """Create a backend instance.

        Returns
        -------
        BaseBackend
            A backend instance
        """
        ...


class BaseBackend(ABC):
    def __init__(self, base_dir: Path):
        """Initialize the backend with the base directory.

        Parameters
        ----------
        base_dir : Path, default="."
            The base directory where files will be created.
        """
        self._base_dir = base_dir.resolve()
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        fn: Callable,
        args_list: Sequence[Any],
    ) -> Any:
        pass

    @abstractmethod
    def execute_impl(
        self,
        fn: SerializedCallable,
        args_list: Sequence[SerializedArgs],
    ) -> Any:
        """Execute a serialized callable with multiple arguments in parallel. In
        Python, this would translate to:

        ```python
        import multiprocessing as mp

        with mp.Pool() as pool:
            results = pool.starmap(fn, args_list)
        ```

        This method should be implemented by subclasses to define how a
        serialized callable should be executed in parallel with multiple sets
        of serialized arguments.

        Parameters
        ----------
        fn : SerializedCallable
            The serialized callable to be executed
        args_list : Sequence[SerializedArgs]
            A sequence of serialized arguments to be processed in parallel

        Returns
        -------
        Any
            The results of executing the callable with the provided arguments

        Notes
        -----
        Implementations that do not support parallel execution should raise an
        exception if `len(args_list) > 1`.
        """
        ...
