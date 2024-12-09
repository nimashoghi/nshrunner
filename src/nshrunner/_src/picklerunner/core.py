from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cloudpickle
from typing_extensions import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T", infer_variance=True)


def save_function(path: Path, fn: Callable[P, T]) -> None:
    """Save a function to a pickle file.

    Parameters
    ----------
    path : Path
        Path to save the pickled function
    fn : Callable
        Function to pickle
    """
    with path.open("wb") as f:
        cloudpickle.dump(fn, f)


def save_args(path: Path, *args: Any, **kwargs: Any) -> None:
    """Save function arguments to a pickle file.

    Parameters
    ----------
    path : Path
        Path to save the pickled arguments
    *args : P.args
        Positional arguments to pickle
    **kwargs : P.kwargs
        Keyword arguments to pickle
    """
    data = {"args": args, "kwargs": kwargs}
    with path.open("wb") as f:
        cloudpickle.dump(data, f)


def save_function_call(
    fn_path: Path,
    args_path: Path,
    fn: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> None:
    """Save both a function and its arguments to separate pickle files.

    Parameters
    ----------
    fn_path : Path
        Path to save the pickled function
    args_path : Path
        Path to save the pickled arguments
    fn : Callable
        Function to pickle
    *args : P.args
        Positional arguments to pickle
    **kwargs : P.kwargs
        Keyword arguments to pickle

    Raises
    ------
    PickleRunnerError
        If there are issues saving the function or arguments
    """
    try:
        save_function(fn_path, fn)
        save_args(args_path, *args, **kwargs)
    except Exception as e:
        raise PickleRunnerError(f"Error saving function and args: {e}") from e


def execute(
    fn_path: Path,
    args_path: Path,
    save_path: Path | None = None,
) -> Any:
    """Execute a pickled function with the given pickled arguments and save the result.

    Parameters
    ----------
    fn_path : Path
        Path to the pickled function
    args_path : Path
        Path to the pickled arguments
    save_path : Path, optional
        Path to save the result, by default None

    Returns
    -------
    Any
        The result of the function call.

    Raises
    ------
    PickleRunnerError
        If there are issues loading, executing, or saving results
    """
    try:
        with fn_path.open("rb") as f:
            fn = cloudpickle.load(f)

        with args_path.open("rb") as f:
            args_data = cloudpickle.load(f)

        result = fn(*args_data["args"], **args_data["kwargs"])

        # Save the result
        if save_path is not None:
            with save_path.open("wb") as f:
                cloudpickle.dump(result, f)

        return result

    except Exception as e:
        raise PickleRunnerError(f"Error executing pickled function: {e}") from e


class PickleRunnerError(Exception):
    """Base exception for pickle runner errors."""


@dataclass(frozen=True)
class SerializedArgs:
    """Represents serialized arguments stored in a pickle file."""

    path: Path

    @classmethod
    def save(cls, path: Path, *args: Any, **kwargs: Any) -> SerializedArgs:
        """Save arguments to a pickle file."""
        save_args(path, *args, **kwargs)
        return cls(path)


@dataclass(frozen=True)
class SerializedCallable:
    """Represents a serialized function stored in a pickle file."""

    path: Path

    @classmethod
    def save(cls, path: Path, fn: Callable) -> SerializedCallable:
        """Save a function to a pickle file."""
        save_function(path, fn)
        return cls(path)


@dataclass(frozen=True)
class PickledFunctionCall:
    """Represents a complete function call that can be executed: a function and its arguments."""

    fn: SerializedCallable
    args: SerializedArgs

    def execute(self, save_path: Path | None = None) -> Any:
        """
        Execute this function call using the pickled function and arguments.

        Parameters
        ----------
        save_path : Path, optional
            Path to save the result, by default None

        Returns
        -------

        Any
            The result of the function call.
        """
        return execute(
            self.fn.path,
            self.args.path,
            save_path=save_path,
        )
