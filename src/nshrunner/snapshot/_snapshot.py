import importlib.util
import logging
import subprocess
import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ._constant import SNAPSHOT_DIR_NAME_DEFAULT

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


def snapshot(
    self,
    snapshot: bool | SnapshotConfig,
    configs: Sequence[Any],
    local_data_path: Path,
):
    # Handle snapshot
    snapshot_config: SnapshotConfig | None = None
    if snapshot is True:
        snapshot_config = {**SNAPSHOT_CONFIG_DEFAULT}
    elif snapshot is False:
        snapshot_config = None
    elif isinstance(snapshot, Mapping):
        snapshot_config = {**SNAPSHOT_CONFIG_DEFAULT, **snapshot}

    del snapshot
    if snapshot_config is None:
        return None

    # Set the snapshot base to the user's home directory
    snapshot_dir = snapshot_config.get("dir", local_data_path / "snapshot")
    snapshot_dir.mkdir(exist_ok=True, parents=True)

    snapshot_modules_set: set[str] = set()
    snapshot_modules_set.update(snapshot_config.get("modules", []))
    if snapshot_config.get("snapshot_ll", True):
        # Resolve ll by taking the module of the runner class
        ll_module = self.__class__.__module__.split(".", 1)[0]
        if ll_module != "ll":
            log.warning(
                f"Runner class {self.__class__.__name__} is not in the 'll' module.\n"
                "This is unexpected and may lead to issues with snapshotting."
            )
        snapshot_modules_set.add(ll_module)
    if snapshot_config.get("snapshot_config_cls_module", True):
        for config in configs:
            # Resolve the root module of the config class
            # NOTE: We also must handle the case where the config
            #   class's module is "__main__" (i.e. the config class
            #   is defined in the main script).
            module = config.__class__.__module__
            if module == "__main__":
                log.warning(
                    f"Config class {config.__class__.__name__} is defined in the main script.\n"
                    "Snapshotting the main script is not supported.\n"
                    "Skipping snapshotting of the config class's module."
                )
                continue

            # Make sure to get the root module
            module = module.split(".", 1)[0]
            snapshot_modules_set.add(module)

    snapshot_path = snapshot_modules(snapshot_dir, list(snapshot_modules_set))
    return snapshot_path.absolute()
