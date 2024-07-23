import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Literal, TypeAlias

import cloudpickle as pickle
from typing_extensions import TypedDict, TypeVarTuple, Unpack, override

from ._types import SerializedFunctionCallDict

_Path: TypeAlias = str | Path | PathLike


@dataclass(frozen=True)
class _SerializedFunction(PathLike):
    path: Path

    _additional_command_parts: Sequence[str] = ()

    def to_command_parts(self, python_executable: str | None = None):
        if python_executable is None:
            python_executable = sys.executable

        return [
            python_executable,
            "-m",
            __name__,
            str(self.path),
            *self._additional_command_parts,
        ]

    def to_command_str(self, python_executable: str | None = None) -> str:
        return " ".join(self.to_command_parts(python_executable))

    @override
    def __fspath__(self) -> str:
        return str(self.path)


def serialize_single(
    dest: _Path,
    fn: Callable,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    additional_command_parts: Sequence[str] = (),
):
    serialized: SerializedFunctionCallDict = {"fn": fn, "args": args, "kwargs": kwargs}

    dest = Path(dest)
    with dest.open("wb") as file:
        pickle.dump(serialized, file)

    return _SerializedFunction(dest, additional_command_parts)


@dataclass(frozen=True)
class SerializedMultiFunction(PathLike):
    base_dir: Path
    functions: Sequence[_SerializedFunction]
    _additional_command_parts: Sequence[str] = ()

    def to_bash_command(
        self,
        job_index_variable: str,
        python_executable: str | None = None,
        environment: Mapping[str, str] | None = None,
        print_environment_info: bool = False,
        pause_before_exit: bool = False,
    ) -> list[str]:
        if python_executable is None:
            python_executable = sys.executable

        command: list[str] = []

        # command = f'{python_executable} -m {__name__} "{str(self.base_dir.absolute())}/${{{job_index_variable}}}.pkl"'
        command.append(python_executable)
        command.append("-m")
        command.append(__name__)

        if environment:
            for key, value in environment.items():
                command.append("--env")
                command.append(f"{key}={value}")

        if print_environment_info:
            command.append("--print-environment-info")

        if pause_before_exit:
            command.append("--pause-before-exit")

        command.append(
            f'"{str(self.base_dir.absolute())}/${{{job_index_variable}}}.pkl"'
        )

        if self._additional_command_parts:
            # command += " " + " ".join(self._additional_command_parts)
            command.extend(self._additional_command_parts)
        return command

    def bash_command_sequential(
        self,
        python_executable: str | None = None,
        environment: Mapping[str, str] | None = None,
        pause_before_exit: bool = False,
        print_environment_info: bool = False,
    ) -> list[str]:
        if python_executable is None:
            python_executable = sys.executable

        all_files = [f'"{str(fn.path.absolute())}"' for fn in self.functions]

        command: list[str] = []
        # command = f"{python_executable} -m {__name__} {all_files}"
        command.append(python_executable)
        command.append("-m")
        command.append(__name__)

        if environment:
            for key, value in environment.items():
                command.append("--env")
                command.append(f"{key}={value}")

        if print_environment_info:
            command.append("--print-environment-info")
        else:
            command.append("--no-print-environment-info")

        if pause_before_exit:
            command.append("--pause-before-exit")

        command.extend(all_files)

        if self._additional_command_parts:
            # command += " " + " ".join(self._additional_command_parts)
            command.extend(self._additional_command_parts)
        return command

    @override
    def __fspath__(self) -> str:
        return str(self.base_dir)


def serialize_many(
    destdir: _Path,
    fn: Callable,
    args_and_kwargs_list: Sequence[tuple[Sequence[Any], Mapping[str, Any]]],
    start_idx: int = 1,
    additional_command_parts: Sequence[str] = (),
):
    serialized_list: list[_SerializedFunction] = []

    destdir = Path(destdir)
    for i, (args, kwargs) in enumerate(args_and_kwargs_list):
        dest = destdir / f"{i+start_idx}.pkl"
        serialized = serialize_single(dest, fn, args, kwargs)
        serialized_list.append(serialized)

    return SerializedMultiFunction(destdir, serialized_list, additional_command_parts)


def _write_helper_script(
    script_path: Path,
    command: str | Iterable[str],
    environment: Mapping[str, str],
    setup_commands: Sequence[str],
    chmod: bool = True,
    prepend_command_with_exec: bool = True,
):
    """
    Creates a helper bash script for running the given function.

    The core idea: The helper script is essentially one additional layer of indirection
    that allows us to encapsulates the environment setup and the actual function call
    in a single bash script (that does not require properly set up Python environment).

    In effect, this allows us to, for example:
    - Easily run the function in the correct environment
        (without having to deal with shell hooks)
        using `conda run -n myenv bash /path/to/helper.sh`.
    - Easily run the function in a Singularity container
        using `singularity exec my_container.sif bash /path/to/helper.sh`.
    """

    with script_path.open("w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("set -e\n\n")

        if environment:
            for key, value in environment.items():
                f.write(f"export {key}={value}\n")
            f.write("\n")

        if setup_commands:
            for setup_command in setup_commands:
                f.write(f"{setup_command}\n")
            f.write("\n")

        if not isinstance(command, str):
            command = " ".join(command)

        if prepend_command_with_exec:
            command = f"exec {command}"
        f.write(f"{command}\n")

    if chmod:
        # Make the script executable
        script_path.chmod(0o755)


TArguments = TypeVarTuple("TArguments")


class SequentialExecutionConfig(TypedDict):
    mode: Literal["sequential"]

    print_environment_info: bool
    """Print the environment information before starting the session."""

    environment: Mapping[str, str]
    """Set the environment variables."""

    python_executable: str | None
    """Python executable to use for running the script."""

    pause_before_exit: bool
    """Wait for the user to press Enter after finishing."""


class ArrayExecutionConfig(TypedDict):
    mode: Literal["array"]

    job_index_variable: str
    """Name of the environment variable that contains the job index. E.g., `SLURM_ARRAY_TASK_ID` for Slurm."""

    print_environment_info: bool
    """Print the environment information before starting the session."""

    environment: Mapping[str, str]
    """Set the environment variables."""

    python_executable: str | None
    """Python executable to use for running the script."""

    pause_before_exit: bool
    """Wait for the user to press Enter after finishing."""


def callable_to_command(
    script_path: Path,
    callable: Callable[[Unpack[TArguments]], Any],
    args_list: Sequence[tuple[Unpack[TArguments]]],
    execution: SequentialExecutionConfig | ArrayExecutionConfig,
    command_template: str = "bash {script}",
):
    # Validate args:
    # - script_path must end with '.sh'
    if not script_path.name.endswith(".sh"):
        raise ValueError("The script path must end with '.sh'.")

    # - `command_template` must contain '{script}'
    if "{script}" not in command_template:
        raise ValueError("The command template must contain '{script}'.")

    from ..picklerunner.create import serialize_many

    # Convert the command/callable to a string for the command
    destdir = script_path.parent / "fns"
    destdir.mkdir(exist_ok=True, parents=True)

    serialized_command = serialize_many(
        destdir,
        callable,
        [(args, {}) for args in args_list],
        start_idx=1,  # Slurm job indices are 1-based
    )
    match execution:
        case {"mode": "sequential", **kwargs}:
            command_inner = serialized_command.bash_command_sequential(**kwargs)
        case {"mode": "array", **kwargs}:
            command_inner = serialized_command.to_bash_command(**kwargs)
        case _:
            raise ValueError(f"Invalid execution mode: {execution['mode']}")

    helper_path = script_path.with_suffix(".helper.sh")
    _write_helper_script(
        helper_path,
        command_inner,
        kwargs.get("environment", {}),
        kwargs.get("setup_commands", []),
        prepend_command_with_exec=True,
    )
    command = command_template.format(script=str(helper_path.absolute()))

    return command