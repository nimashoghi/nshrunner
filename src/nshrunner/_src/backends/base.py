from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..picklerunner import SerializedArgs, SerializedCallable


@dataclass(frozen=True)
class JobInfo:
    id: str
    """The job ID."""


@dataclass(frozen=True)
class JobStatus:
    """Represents the status of a job."""

    status: str
    """The job status."""

    description: str = ""
    """Any additional information about the job status."""


@runtime_checkable
class Job(Protocol):
    def info(self) -> JobInfo:
        """Get information about the job.

        Returns
        -------
        JobInfo
            The job information
        """
        ...

    def status(self) -> JobStatus:
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


class BaseBackend(ABC):
    @abstractmethod
    def execute_parallel(
        self,
        fn: SerializedCallable,
        args: Sequence[SerializedArgs],
    ) -> Any:
        """Execute a serialized callable with multiple arguments in parallel.

        This method should be implemented by subclasses to define how a
        serialized callable should be executed in parallel with multiple sets
        of serialized arguments.

        Parameters
        ----------
        fn : SerializedCallable
            The serialized callable to be executed
        args : Sequence[SerializedArgs]
            A sequence of serialized arguments to be processed in parallel

        Returns
        -------
        Any
            The results of executing the callable with the provided arguments
        """
        ...

    @abstractmethod
    def execute_single(
        self,
        fn: SerializedCallable,
        args: SerializedArgs,
    ) -> Any:
        """Execute a serialized callable with a single set of arguments.

        This method should be implemented by subclasses to define how a
        serialized callable should be executed with a single set of
        serialized arguments.

        Parameters
        ----------
        fn : SerializedCallable
            The serialized callable to be executed
        args : SerializedArgs
            The serialized arguments to pass to the callable

        Returns
        -------
        Any
            The result of executing the callable with the provided arguments
        """
        ...

    @abstractmethod
    def execute_sequential(
        self,
        fn: SerializedCallable,
        args: Sequence[SerializedArgs],
    ) -> Any:
        """Execute a serialized callable with multiple arguments sequentially.

        This method should be implemented by subclasses to define how a
        serialized callable should be executed sequentially with multiple sets
        of serialized arguments.

        Parameters
        ----------
        fn : SerializedCallable
            The serialized callable to be executed
        args : Sequence[SerializedArgs]
            A sequence of serialized arguments to be processed sequentially

        Returns
        -------
        Any
            The results of executing the callable with the provided arguments
        """
        ...
