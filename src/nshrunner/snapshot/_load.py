import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Literal

from typing_extensions import assert_never

log = logging.getLogger(__name__)


def load_existing_snapshot(
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

    snapshot_dir = snapshot_dir.absolute()
    snapshot_dir_str = str(snapshot_dir)
    # If the snapshot directory is already in the Python path, do nothing
    if snapshot_dir_str in sys.path:
        log.info(f"Snapshot directory {snapshot_dir} already in sys.path")
        return

    # Iterate through all the modules within the snapshot directory
    modules_list: list[str] = []
    errors: list[str] = []
    for module_dir in snapshot_dir.iterdir():
        module_dir = module_dir.absolute()
        if not module_dir.is_dir():
            continue

        # Check if the module exists in the filesystem
        if (spec := importlib.util.find_spec(module_dir.name)) is not None:
            log.debug(
                f"Module {module_dir.name} exists in the filesystem. Path: {spec.origin}"
            )
            if spec.origin:
                original_path = Path(spec.origin).absolute()
                log.info(f"Module {module_dir.name}: {original_path} -> {module_dir}")

        # If the module has already been imported, warn the user
        if module_dir.name in sys.modules:
            errors.append(
                f"Module {module_dir.name} has already been imported. "
                "All previously imported modules will not be updated."
            )
            continue

        modules_list.append(module_dir.name)

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
