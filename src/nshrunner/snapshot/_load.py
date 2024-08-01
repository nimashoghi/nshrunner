import contextlib
import importlib
import importlib.util
import logging
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path
from typing import Literal, TypeAlias

from typing_extensions import assert_never, final, override

log = logging.getLogger(__name__)


ACTIVE_SNAPSHOT_ENV_VAR = "NSHRUNNER_ACTIVE_SNAPSHOT_DIR"

OnErrorType: TypeAlias = Literal["warn", "raise"]
OnExistingSnapshotType: TypeAlias = Literal[
    "warn_and_overwrite",
    "warn_and_ignore",
    "raise",
]


@final
class LoadExistingSnapshotContext(contextlib.AbstractContextManager):
    snapshot_dirs: list[Path]
    on_existing_snapshot: OnExistingSnapshotType

    @override
    def __init__(
        self,
        snapshot_dirs: list[Path],
        on_existing_snapshot: OnExistingSnapshotType,
        remove_paths: list[Path] = [],
    ):
        super().__init__()

        self.snapshot_dirs = [dir.absolute() for dir in snapshot_dirs]
        self.on_existing_snapshot = on_existing_snapshot
        self.remove_paths = remove_paths

        self.__enter__()

    @staticmethod
    def _get_snapshot_dirs() -> list[str]:
        """Retrieve and parse snapshot directories from the environment variable."""
        raw_value = os.environ.get(ACTIVE_SNAPSHOT_ENV_VAR, "")
        return [s for s in raw_value.split(os.pathsep) if s]

    @staticmethod
    def _set_snapshot_dirs(dirs: list[str]) -> None:
        """Set the environment variable with the given directories."""
        os.environ[ACTIVE_SNAPSHOT_ENV_VAR] = os.pathsep.join(dirs)

    @staticmethod
    def _clear_snapshot_dirs() -> None:
        """Remove the environment variable."""
        os.environ.pop(ACTIVE_SNAPSHOT_ENV_VAR, None)

    def _load_snapshots(
        self, snapshot_dir_strs: list[str], reset_import_cache: bool = True
    ):
        for snapshot_dir_str in snapshot_dir_strs:
            sys.path.insert(0, snapshot_dir_str)
        self._set_snapshot_dirs(snapshot_dir_strs)
        log.info(f"Added {', '.join(snapshot_dir_strs)} to sys.path.")

        # Reset the import cache to ensure that the new modules are imported
        if reset_import_cache:
            importlib.invalidate_caches()

    def _unload_snapshots(
        self, snapshot_dir_strs: list[str], reset_import_cache: bool = True
    ):
        existing_snapshots = self._get_snapshot_dirs()
        if set(existing_snapshots) != set(snapshot_dir_strs):
            raise RuntimeError(
                f"Request to unload snapshot directories {snapshot_dir_strs} is invalid. "
                f"Active snapshot directories are {existing_snapshots}."
            )

        # Remove from sys path, or raise an error if it's not there
        for snapshot_dir_str in snapshot_dir_strs:
            try:
                sys.path.remove(snapshot_dir_str)
            except ValueError:
                raise RuntimeError(
                    f"Snapshot directory {snapshot_dir_str} not found in sys.path."
                )

        self._clear_snapshot_dirs()

        # Reset the import cache to ensure that the new modules are imported
        if reset_import_cache:
            importlib.invalidate_caches()

    @override
    def __enter__(self):
        snapshot_dir_strs = [str(dir) for dir in self.snapshot_dirs]

        # Check to see if there are existing snapshots
        if existing_snapshots := self._get_snapshot_dirs():
            log.info(f"Existing snapshot directories: {', '.join(existing_snapshots)}")
            match self.on_existing_snapshot:
                case "warn_and_overwrite":
                    log.warning(
                        f"Other snapshot directories {', '.join(existing_snapshots)} are already active. "
                        f"Overwriting with {', '.join(snapshot_dir_strs)}"
                    )
                case "warn_and_ignore":
                    log.warning(
                        f"Other snapshot directories {', '.join(existing_snapshots)} are already active. "
                        f"Ignoring {', '.join(snapshot_dir_strs)}"
                    )
                    return
                case "raise":
                    raise RuntimeError(
                        f"Other snapshot directories {', '.join(existing_snapshots)} are already active. "
                        f"Cannot load {', '.join(snapshot_dir_strs)}"
                    )
                case _:
                    assert_never(self.on_existing_snapshot)

            # If we made it here, we should overwrite the existing snapshots
            self._unload_snapshots(existing_snapshots, reset_import_cache=False)

        # Add the snapshot directories to the Python path
        self._load_snapshots(snapshot_dir_strs)

    @override
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> bool | None:
        # Unload the current snapshot directories
        if not (existing_snapshots := self._get_snapshot_dirs()):
            raise RuntimeError("No snapshot directories are active.")
        self._unload_snapshots(existing_snapshots)

        # Remove directories that were created
        for p in self.remove_paths:
            if not p.exists():
                continue

            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
            else:
                log.warn(
                    f"Path {p} is not a file, symlink, or directory. "
                    "Not sure how to remove it."
                )


def _validate_snapshot(snapshot_dir: Path, modules_list: list[tuple[str, Path]]):
    # In this case, we can just add the snapshot directory to the Python path,
    # because all modules should be direct children of the snapshot directory.
    # Let's validate that assumption
    for name, module_dir in modules_list:
        if module_dir.parent != snapshot_dir:
            raise RuntimeError(
                f"Module {name} is not a direct child of the snapshot directory {snapshot_dir}. "
                "This should not happen when `preserve_original_modules` is False. "
                "Please report this as a bug."
            )

    log.critical(
        f"Loading the following modules from {snapshot_dir}: {', '.join(name for name, _ in modules_list)}"
    )


def _copy_modules_to_temp_dir(modules_list: list[tuple[str, Path]]):
    # Create a temporary directory to store the original modules
    dir_ = Path(tempfile.mkdtemp(prefix="nshrunner_snapshot_original_modules_"))

    # Copy the original modules to the temporary directory
    new_modules_list: list[tuple[str, Path]] = []
    for name, module_dir in modules_list:
        target_dir = dir_ / f"{name}_original"

        # Preferably, we would like to use a symlink.
        try:
            target_dir.symlink_to(module_dir)
        except OSError:
            # If we can't use a symlink, we will copy the directory instead
            shutil.copytree(module_dir, target_dir)
            log.info(f"Copied {module_dir} to {target_dir}")
        else:
            log.info(f"Symlinked {module_dir} to {target_dir}")

        new_modules_list.append((name, target_dir))

    return dir_, new_modules_list


def _enter_snapshot(
    snapshot_dir: Path,
    modules_list_snapshot: list[tuple[str, Path]],
    modules_list_original: list[tuple[str, Path]],
    on_existing_snapshot: OnExistingSnapshotType,
):
    if not modules_list_original:
        _validate_snapshot(snapshot_dir, modules_list_snapshot)
        return LoadExistingSnapshotContext([snapshot_dir], on_existing_snapshot)

    # Otherwise, we need to create a new temporary directory to store the original modules
    # and then add the snapshot directory to the Python path.
    # We will then move the original modules to the temporary directory.
    original_dir, modules_list_original = _copy_modules_to_temp_dir(
        modules_list_original
    )

    for modules_dir, modules_list in zip(
        (snapshot_dir, original_dir),
        (modules_list_snapshot, modules_list_original),
    ):
        _validate_snapshot(modules_dir, modules_list)

    return LoadExistingSnapshotContext(
        [snapshot_dir, original_dir],
        on_existing_snapshot,
        remove_paths=[original_dir],
    )


def load_existing_snapshot(
    snapshot_dir: Path,
    *,
    on_error: OnErrorType = "raise",
    on_existing_snapshot: OnExistingSnapshotType = "raise",
    preserve_original_modules: bool = False,
):
    """
    Add the snapshot directory to PYTHONPATH.

    If `preserve_original_modules` is True, we will store the original module
    as `{module_name}_original`, so the user can access the original module
    if needed. This lets you mix-and-match between the original and snapshot
    modules. Note that this can lead to unexpected behavior if the original
    module uses absolute imports for its own submodules, as this will import
    the snapshot submodules instead.

    Warns on:
    - Modules within the snapshot directory that have already been imported
        (and thus any previously imported module will not be updated).
    """

    snapshot_dir = snapshot_dir.absolute()

    # Iterate through all the modules within the snapshot directory
    modules_list_snapshot: list[tuple[str, Path]] = []
    modules_list_original: list[tuple[str, Path]] = []
    errors: list[str] = []
    for module_dir in snapshot_dir.iterdir():
        if not module_dir.is_dir():
            continue

        module_dir = module_dir.absolute()

        # Check if the module exists in the filesystem
        if (spec := importlib.util.find_spec(module_dir.name)) is not None:
            log.debug(
                f"Module {module_dir.name} exists in the filesystem. Path: {spec.origin}"
            )
            if spec.origin:
                # Get the original path of the module
                original_path = Path(spec.origin).parent.absolute()
                log.info(f"Module {module_dir.name}: {original_path} -> {module_dir}")

                if preserve_original_modules:
                    # Store the original module so we can save it
                    # for loading the original module later.
                    modules_list_original.append((module_dir.name, original_path))

        # If the module has already been imported, warn the user
        if module_dir.name in sys.modules:
            errors.append(
                f"Module {module_dir.name} has already been imported. "
                "All previously imported modules will not be updated."
            )
            continue

        modules_list_snapshot.append((module_dir.name, module_dir))

    # If there are any errors, handle them according to the `on_error` parameter
    if errors:
        match on_error:
            case "warn":
                log.warning("\n".join(errors))
            case "raise":
                raise RuntimeError("\n".join(errors))
            case _:
                assert_never(on_error)

    # Enter the snapshot context
    return _enter_snapshot(
        snapshot_dir,
        modules_list_snapshot=modules_list_snapshot,
        modules_list_original=modules_list_original,
        on_existing_snapshot=on_existing_snapshot,
    )
