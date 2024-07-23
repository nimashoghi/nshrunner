import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, TypeAlias

import cloudpickle as pickle
from typing_extensions import override

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
    start_idx: int = 0,
    additional_command_parts: Sequence[str] = (),
):
    serialized_list: list[_SerializedFunction] = []

    destdir = Path(destdir)
    for i, (args, kwargs) in enumerate(args_and_kwargs_list):
        dest = destdir / f"{i+start_idx}.pkl"
        serialized = serialize_single(dest, fn, args, kwargs)
        serialized_list.append(serialized)

    return SerializedMultiFunction(destdir, serialized_list, additional_command_parts)
