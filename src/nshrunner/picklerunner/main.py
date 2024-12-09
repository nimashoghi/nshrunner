from __future__ import annotations

import argparse
import contextlib
import logging
import os
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import cast

import cloudpickle
from typing_extensions import TypeAliasType

from ._util import SerializedFunctionCallDict

_Path = TypeAliasType("_Path", str | Path | PathLike)


def _execute_single(path: _Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path {path} does not exist")

    with path.open("rb") as file:
        d = cloudpickle.load(file)

    # Validate the dict.
    assert isinstance(d, Mapping), f"Expected a dict, got {type(d)}"
    d = cast(SerializedFunctionCallDict, d)
    # `fn`
    assert (fn := d.get("fn")) is not None, f"Expected a 'fn' key, got {d.keys()}"
    assert callable(fn), f"Expected a callable, got {type(fn)}"
    # `args`
    assert (args := d.get("args")) is not None, f"Expected a 'args' key, got {d.keys()}"
    assert isinstance(args, Sequence), f"Expected a tuple, got {type(args)}"
    # `kwargs`
    assert (
        kwargs := d.get("kwargs")
    ) is not None, f"Expected a 'kwargs' key, got {d.keys()}"
    assert isinstance(kwargs, Mapping), f"Expected a dict, got {type(kwargs)}"

    # Call the function and return the result.
    return fn(*args, **kwargs)


def _resolve_paths(paths: Sequence[Path]):
    for path in paths:
        if path.is_file():
            yield path
            continue

        for child in path.iterdir():
            if child.is_file() and child.suffix == ".pkl":
                yield child


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="Paths to the sessions to run",
    )
    parser.add_argument(
        "--print-environment-info",
        action=argparse.BooleanOptionalAction,
        help="Print the environment information before starting the session",
        default=False,
    )
    parser.add_argument(
        "--env",
        "-e",
        help="Set the environment variable. Format: KEY=VALUE",
        action="append",
    )
    parser.add_argument(
        "--replace-rocr-visible-devices",
        action=argparse.BooleanOptionalAction,
        help="Replace the ROCR_VISIBLE_DEVICES environment variable with CUDA_VISIBLE_DEVICES",
        default=True,
    )
    parser.add_argument(
        "--pause-before-exit",
        action=argparse.BooleanOptionalAction,
        help="Wait for the user to press Enter after finishing",
        default=False,
    )

    args = parser.parse_args()
    return args


@contextlib.contextmanager
def _set_env(key: str, value: str):
    original_value = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if original_value is not None:
            os.environ[key] = original_value
        else:
            del os.environ[key]


def _to_result_path(script_path: Path):
    results_dir = script_path.parent / "results"
    results_dir.mkdir(exist_ok=True)

    i = 0
    while (fpath := results_dir / f"{script_path.stem}.rank{i}.pkl").exists():
        i += 1

    return fpath


def main():
    with contextlib.ExitStack() as stack:
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger(__name__)
        args = _parse_args()

        # Print the environment information if requested.
        if args.print_environment_info:
            from .._util.print_environment_info import print_environment_info

            print_environment_info(log)

        # Set the environment variables if requested.
        if args.env:
            for env in args.env:
                key, value = env.split("=", 1)
                log.critical(f"Setting {key}={value}...")
                stack.enter_context(_set_env(key, value))

        # Replace the ROCR_VISIBLE_DEVICES environment variable with CUDA_VISIBLE_DEVICES if requested.
        if args.replace_rocr_visible_devices:
            if "ROCR_VISIBLE_DEVICES" in os.environ:
                log.critical(
                    "Replacing ROCR_VISIBLE_DEVICES with CUDA_VISIBLE_DEVICES..."
                )
                os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["ROCR_VISIBLE_DEVICES"]
                os.environ.pop("ROCR_VISIBLE_DEVICES")

        if not (paths := list(_resolve_paths(args.paths))):
            raise ValueError("No paths provided")

        # Sort by the job index.
        paths = sorted(paths, key=lambda path: int(path.stem))

        # Make sure all paths exist
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Path {path} does not exist")

        # Execute the sessions.
        for i, path in enumerate(paths):
            log.critical(f"Executing #{i}: {path=}...")
            result = _execute_single(path)
            result_path = _to_result_path(path)
            log.critical(f"Saving {result=} to {result_path}...")
            with result_path.open("wb") as file:
                cloudpickle.dump(result, file)

        log.critical("Finished running all sessions.")

        if args.pause_before_exit:
            # input("Press Enter to continue...")
            # Enter causes issues when we wrap this in a script.
            # Instead, we'll just wait for the user to ctrl+c.
            log.critical("Execution complete. Press Ctrl+C to exit...")
            try:
                # Use a simple loop that can be interrupted
                while True:
                    import time

                    time.sleep(1)
            except KeyboardInterrupt:
                log.critical("Received interrupt, exiting...")

        log.critical("Exiting...")


if __name__ == "__main__":
    main()
