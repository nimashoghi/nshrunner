import importlib.util
import logging
import subprocess
import sys
import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from typing_extensions import assert_never

log = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SnapshotInformation:
    snapshot_dir: Path
    moved_modules: dict[str, list[tuple[Path, Path]]]


def _copy(source: Path, location: Path):
    """
    Copy files from the source directory to the specified location, excluding ignored files.

    Args:
        source (Path): The path to the source directory.
        location (Path): The path to the destination directory.

    Raises:
        CalledProcessError: If the rsync command fails.

    """
    ignored_files = (
        subprocess.check_output(
            [
                "git",
                "-C",
                str(source),
                "ls-files",
                "--exclude-standard",
                "-oi",
                "--directory",
            ]
        )
        .decode("utf-8")
        .splitlines()
    )

    # run rsync with .git folder and `ignored_files` excluded
    _ = subprocess.run(
        [
            "rsync",
            "-a",
            "--exclude",
            ".git",
            *(f"--exclude={file}" for file in ignored_files),
            str(source),
            str(location),
        ],
        check=True,
    )


def resolve_snapshot_dir(
    base: str | Path,
    id: str | None = None,
    add_date_to_dir: bool = True,
    error_on_existing: bool = True,
) -> Path:
    """
    Resolve the directory path for a snapshot.

    Args:
        base (str | Path): The base directory path.
        id (str | None, optional): The ID of the snapshot. If None, a new UUID will be generated. Defaults to None.
        add_date_to_dir (bool, optional): Whether to add the current date to the directory path. Defaults to True.
        error_on_existing (bool, optional): Whether to raise an error if the directory already exists. Defaults to True.

    Returns:
        Path: The resolved directory path for the snapshot.
    """
    if id is None:
        id = str(uuid.uuid4())

    snapshot_dir = Path(base)
    if add_date_to_dir:
        snapshot_dir = snapshot_dir / datetime.now().strftime("%Y-%m-%d")
    snapshot_dir = snapshot_dir / id
    snapshot_dir.mkdir(parents=True, exist_ok=not error_on_existing)
    return snapshot_dir


SNAPSHOT_DIR_NAME_DEFAULT = "ll_snapshot"


def snapshot_modules(
    snapshot_dir: Path,
    modules: Sequence[str],
    *,
    snapshot_dir_name: str = SNAPSHOT_DIR_NAME_DEFAULT,
):
    """
    Snapshot the specified modules to the given directory.

    Args:
        snapshot_dir (Path): The directory where the modules will be snapshot.
        modules (Sequence[str]): A sequence of module names to be snapshot.

    Returns:
        Path: The path to the snapshot directory.

    Raises:
        AssertionError: If a module is not found or if a module has a non-directory location.
    """
    snapshot_dir = snapshot_dir / snapshot_dir_name
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    log.critical(f"Snapshotting {modules=} to {snapshot_dir}")

    moved_modules = defaultdict[str, list[tuple[Path, Path]]](list)
    for module in modules:
        spec = importlib.util.find_spec(module)
        if spec is None:
            log.warning(f"Module {module} not found")
            continue

        assert (
            spec.submodule_search_locations
            and len(spec.submodule_search_locations) == 1
        ), f"Could not find module {module} in a single location."
        location = Path(spec.submodule_search_locations[0])
        assert (
            location.is_dir()
        ), f"Module {module} has a non-directory location {location}"

        (*parent_modules, module_name) = module.split(".")

        destination = snapshot_dir
        for part in parent_modules:
            destination = destination / part
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "__init__.py").touch(exist_ok=True)

        _copy(location, destination)

        destination = destination / module_name
        log.info(f"Moved {location} to {destination} for {module=}")
        moved_modules[module].append((location, destination))

    return snapshot_dir


def add_snapshot_to_python_path(
    snapshot_dir: Path,
    *,
    on_error: Literal["warn", "raise"] = "raise",
):
    """
    Add the snapshot directory to PYTHONPATH.

    Warns on:
    - Modules within the snapshot directory that have already been imported
        (and thus any previously imported module will not be updated).
    """

    snapshot_dir = snapshot_dir.resolve().absolute()
    snapshot_dir_str = str(snapshot_dir)
    # If the snapshot directory is already in the Python path, do nothing
    if snapshot_dir_str in sys.path:
        log.info(f"Snapshot directory {snapshot_dir} already in sys.path")
        return

    # Iterate through all the modules within the snapshot directory
    modules_list: list[str] = []
    errors: list[str] = []
    for module_dir in snapshot_dir.iterdir():
        if not module_dir.is_dir():
            continue

        module_name = module_dir.name
        # If the module has already been imported, warn the user
        if module_name in sys.modules:
            errors.append(
                f"Module {module_name} has already been imported. "
                "All previously imported modules will not be updated."
            )
            continue

        modules_list.append(module_name)

    # If there are any errors, handle them according to the `on_error` parameter
    if errors:
        match on_error:
            case "warn":
                log.warning("\n".join(errors))
            case "raise":
                raise RuntimeError("\n".join(errors))
            case _:
                assert_never(on_error)

    # Add the snapshot directory to the Python path
    sys.path.insert(0, snapshot_dir_str)
    log.critical(
        f"Added {snapshot_dir} to sys.path. Modules: {', '.join(modules_list)}"
    )

    # Reset the import cache to ensure that the new modules are imported
    importlib.invalidate_caches()


def load_python_path_from_run(run_dir: Path):
    import yaml

    if (hparams_path := next((run_dir / "log").glob("**/hparams.yaml"), None)) is None:
        raise FileNotFoundError(f"Could not find hparams.yaml in {run_dir}")

    config = yaml.unsafe_load(hparams_path.read_text())

    # Find the ll_snapshot if it exists
    if (
        snapshot_path := next(
            (
                path
                for path in config.get("environment", {}).get("python_path", [])
                if path.stem == SNAPSHOT_DIR_NAME_DEFAULT and path.is_dir()
            ),
            None,
        )
    ) is None:
        return

    # Add it to the current python path
    snapshot_path = Path(snapshot_path).absolute()
    add_snapshot_to_python_path(snapshot_path)
