from __future__ import annotations

import inspect
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from git import GitCommandError, InvalidGitRepositoryError, NoSuchPathError, Repo
from nshsnap import SnapshotInfo

from .git import gitignored_dir

log = logging.getLogger(__name__)


def _get_executed_notebook_code() -> str | None:
    """
    Get all executed code cells from the current Jupyter notebook session.

    Returns:
        A string containing all executed code cells formatted with # %% cell delimiters,
        or None if not in an IPython environment or if there's no history.
    """
    try:
        # This will raise NameError if not in IPython
        ipython = get_ipython()  # type: ignore

        if ipython and hasattr(ipython, "history_manager"):
            log.debug("Getting executed code cells from IPython history")

            # List to store all executed code cells
            executed_code_cells = []

            # Loop through the history (session=0 is current session)
            for (
                session_id,
                line_number,
                input_code,
            ) in ipython.history_manager.get_range(session=0):
                # Skip empty cells
                if input_code.strip():
                    executed_code_cells.append(input_code)

            if executed_code_cells:
                log.debug(f"Found {len(executed_code_cells)} executed code cells")

                # Format the cells with # %% delimiters, but avoid duplicating them
                formatted_cells = []
                for code in executed_code_cells:
                    # Check if the cell already starts with # %%
                    if code.lstrip().startswith("# %%"):
                        log.debug("Cell already has # %% delimiter")
                        formatted_cells.append(code)
                    else:
                        log.debug("Adding # %% delimiter to cell")
                        formatted_cells.append(f"# %%\n{code}")

                formatted_notebook = "\n\n".join(formatted_cells)
                return formatted_notebook
            else:
                log.debug("No executed code cells found in history")
                return None
        else:
            log.debug("IPython shell has no history manager")
            return None
    except (NameError, AttributeError) as e:
        log.debug(f"Not in IPython environment or error accessing history: {e}")
        return None


def _get_main_script() -> tuple[Path | None, Literal["script", "notebook"] | None]:
    """
    Detect the main script or notebook that's being executed.

    Returns:
        A tuple of (file_path, file_type) where file_type is either "script" or "notebook".
        If no main script can be detected, returns (None, None).
    """
    log.debug("Attempting to detect main script or notebook")

    # First check if we're in a Jupyter notebook
    try:
        # This will raise NameError if not in IPython
        ipython = get_ipython()  # type: ignore

        log.debug("Running in IPython environment")

        if ipython and hasattr(ipython, "config"):
            # Check if this is a notebook or IPython console
            if getattr(ipython, "kernel", None) is not None:
                log.debug("Running in notebook environment with kernel")
                return (
                    None,
                    "notebook",
                )  # We'll generate the content from history when saving
    except (NameError, AttributeError) as e:
        log.debug(f"Not running in IPython environment: {e}")

    # If not in a notebook, try to get the main Python script
    try:
        main_script = sys.argv[0]
        log.debug(f"sys.argv[0]: {main_script}")

        if main_script and main_script != "-c":
            script_path = Path(main_script)
            log.debug(
                f"Script path: {script_path}, exists: {script_path.exists()}, is_file: {script_path.is_file() if script_path.exists() else 'N/A'}"
            )

            if script_path.exists() and script_path.is_file():
                # Check if it's a Python script
                if script_path.suffix.lower() in (".py", ".pyc", ".pyw"):
                    log.debug(f"Detected Python script: {script_path}")
                    return script_path, "script"
                # Check if it's a notebook
                elif script_path.suffix.lower() == ".ipynb":
                    log.debug(f"Detected Jupyter notebook: {script_path}")
                    return script_path, "notebook"
                else:
                    log.debug(
                        f"Script has unrecognized extension: {script_path.suffix}"
                    )
            else:
                log.debug(f"Script path does not exist or is not a file: {script_path}")
        else:
            log.debug("sys.argv[0] is empty or '-c', can't use for script detection")
    except (IndexError, ValueError) as e:
        log.debug(f"Error getting script from sys.argv: {e}")

    # Fallback: try to find the calling script by inspecting the stack
    try:
        log.debug("Trying to find script by inspecting stack frames")
        frame = inspect.currentframe()
        frame_count = 0

        while frame:
            frame_count += 1
            module_name = frame.f_globals.get("__name__", "unknown")
            file_path = frame.f_globals.get("__file__")

            log.debug(f"Frame {frame_count}: module={module_name}, file={file_path}")

            if file_path:
                path = Path(file_path)
                if path.exists() and path.is_file():
                    if path.suffix.lower() in (".py", ".pyc", ".pyw"):
                        log.debug(f"Found Python script in frame {frame_count}: {path}")
                        return path, "script"
                    elif path.suffix.lower() == ".ipynb":
                        log.debug(
                            f"Found Jupyter notebook in frame {frame_count}: {path}"
                        )
                        return path, "notebook"

            frame = frame.f_back

            # Avoid infinite loops or excessive frame traversal
            if frame_count > 50:
                log.debug("Stopping stack inspection after 50 frames")
                break
    except Exception as e:
        log.debug(f"Error inspecting stack frames: {e}")

    log.debug("Failed to detect main script or notebook")
    return None, None


def _save_git_diff(code_dir: Path) -> Path | None:
    """
    Save the git diff to the code directory if in a git repository.

    Args:
        code_dir: The code directory where the git diff should be saved

    Returns:
        Path to the saved git diff file if successful, None otherwise
    """
    cwd = Path.cwd()

    try:
        # Create a single Repo instance to use for all Git operations
        repo = Repo(cwd, search_parent_directories=True)
        repo_root = Path(repo.working_dir)
        log.debug(f"Detected Git repository at {repo_root}")

        # Get the diff for tracked files using the same repo instance
        tracked_diff = repo.git.diff("HEAD")

        # Get untracked files using the same repo instance
        untracked_files = repo.untracked_files

        # Process untracked files
        combined_untracked_diff = []
        if untracked_files:
            log.debug(f"Found {len(untracked_files)} untracked files")

            # Process untracked files in batches to maintain performance
            BATCH_SIZE = 100
            for i in range(0, len(untracked_files), BATCH_SIZE):
                batch = untracked_files[i : i + BATCH_SIZE]
                log.debug(f"Processing batch of {len(batch)} untracked files")

                try:
                    # For untracked files, diff against /dev/null
                    for file_path in batch:
                        full_path = repo_root / file_path
                        # Only include the file if it exists and is a regular file
                        if full_path.exists() and full_path.is_file():
                            try:
                                file_diff = repo.git.diff(
                                    "--no-index",
                                    "--",
                                    "/dev/null",
                                    str(file_path),
                                    check=False,
                                )

                                # Add to our collection
                                if file_diff:
                                    combined_untracked_diff.append(file_diff)
                            except GitCommandError as e:
                                # GitCommandError with return code 1 is expected for files with differences
                                if e.status == 1 and e.stdout:
                                    combined_untracked_diff.append(e.stdout)
                                else:
                                    log.debug(
                                        f"Error getting diff for {file_path}: {e}"
                                    )
                except Exception as e:
                    log.debug(f"Error processing batch of untracked files: {e}")
                    # Continue with the next batch

        # Combine all diffs
        combined_diff = tracked_diff or ""
        if combined_untracked_diff:
            untracked_diff_output = "\n".join(combined_untracked_diff)
            if combined_diff and untracked_diff_output:
                combined_diff += "\n" + untracked_diff_output
            else:
                combined_diff = untracked_diff_output

        # Create diff file path
        diff_file = code_dir / "git_diff.patch"

        # Always create the file for valid repositories, even if there are no changes
        combined_diff = combined_diff.strip()
        if combined_diff:
            log.debug("Repository has changes, saving git diff")
            with open(diff_file, "w") as f:
                f.write(combined_diff)
            log.debug(f"Saved Git diff to {diff_file}")
        else:
            log.debug("Repository is clean, creating empty git diff file")
            # Create an empty file
            diff_file.touch()
            log.debug(f"Created empty git diff file at {diff_file}")

        return diff_file

    except (InvalidGitRepositoryError, NoSuchPathError, GitCommandError) as e:
        log.debug(f"Not in a Git repository or error accessing Git: {e}")
        return None
    except Exception as e:
        log.debug(f"Unexpected error processing Git repository: {e}")
        return None


def resolve_code_directory(session_dir: Path) -> Path:
    """
    Resolves (and creates if not exists) a code directory structure for the run.

    Args:
        session_dir: The directory for the current session

    Returns:
        The path to the created code directory
    """
    # Create the code directory
    code_dir = gitignored_dir(session_dir / "code", create=True)
    log.debug(f"Created code directory at {code_dir}")
    return code_dir


def _create_snapshot_copy(snapshot: SnapshotInfo, code_dir: Path) -> Path | None:
    """
    Copy the directory that contains snapshot modules.

    Args:
        snapshot: Snapshot information containing the directory
        code_dir: The code directory where snapshot should be copied

    Returns:
        The path to the copied snapshot directory if successful, None otherwise
    """
    if not (snapshot_dir_path := Path(snapshot.snapshot_dir)).exists():
        log.debug(f"Snapshot directory {snapshot_dir_path} does not exist")
        return None

    snapshot_copy_path = code_dir / "snapshot"
    # Check if directory already exists at the destination and remove it if necessary
    if snapshot_copy_path.exists():
        try:
            log.debug(f"Removing existing directory at {snapshot_copy_path}")
            if snapshot_copy_path.is_symlink():
                snapshot_copy_path.unlink(missing_ok=True)
            else:
                shutil.rmtree(snapshot_copy_path, ignore_errors=True)
        except Exception as e:
            log.debug(f"Error removing existing directory: {e}")
            return None

    try:
        log.debug(
            f"Copying snapshot directory from {snapshot_dir_path} to {snapshot_copy_path}"
        )
        # Copy the entire directory structure
        shutil.copytree(snapshot_dir_path, snapshot_copy_path)
        log.debug(f"Copied snapshot directory to {snapshot_copy_path}")
        return snapshot_copy_path
    except Exception as e:
        log.debug(f"Error copying snapshot directory: {e}")
        return None


def _save_main_script_file(
    code_dir: Path,
) -> tuple[Path | None, Literal["script", "notebook"] | None]:
    """
    Save the main script or notebook to the code directory.

    Args:
        code_dir: The code directory where the script should be saved

    Returns:
        A tuple of (saved_file_path, file_type) if successful, (None, None) otherwise
    """
    script_path, script_type = _get_main_script()

    # Special handling for notebooks - use IPython history to create a Python script
    if script_type == "notebook" and script_path is None:
        log.debug("Creating notebook script from IPython history")
        notebook_code = _get_executed_notebook_code()

        if notebook_code:
            try:
                # Save as main_notebook.py
                specific_file = code_dir / "main_notebook.py"
                with open(specific_file, "w") as f:
                    f.write(notebook_code)

                # Also save as main.py for consistency
                main_file = code_dir / "main.py"
                with open(main_file, "w") as f:
                    f.write(notebook_code)

                log.debug(
                    f"Successfully saved executed notebook code to {specific_file} and {main_file}"
                )
                return specific_file, "notebook"
            except Exception as e:
                log.debug(f"Error saving notebook code: {e}")
                return None, None
        else:
            log.debug("No notebook code could be extracted from IPython history")
            return None, None

    # Standard file copying for scripts
    if script_path is None or script_type is None:
        log.debug("No script or notebook detected to save")
        return None, None

    log.debug(f"Detected {script_type}: {script_path}")

    try:
        # Save as main_script.py or similar
        specific_file = code_dir / f"main_{script_type}{script_path.suffix}"
        shutil.copy2(script_path, specific_file)

        # Also save as main.py for consistency
        main_file = code_dir / "main.py"
        shutil.copy2(script_path, main_file)

        log.debug(
            f"Successfully saved {script_type} to {specific_file} and {main_file}"
        )
        return specific_file, script_type
    except Exception as e:
        log.debug(f"Error saving {script_type}: {e}")
        return None, None


@dataclass(frozen=True)
class CodeDirectoryResult:
    code_dir: Path
    """
    Path to the created code directory.
    """

    saved_script_path: Path | None
    """
    Path to the saved script or notebook, if applicable.
    """

    script_type: Literal["script", "notebook"] | None
    """
    Type of the saved script: "script" or "notebook".
    """

    git_diff_path: Path | None
    """
    Path to the saved git diff file, if applicable.
    """


def setup_code_directory(
    session_dir: Path,
    snapshot: SnapshotInfo | None,
    save_main_script: bool = True,
    save_git_diff: bool = True,
) -> CodeDirectoryResult:
    """
    Create the code directory and save relevant code artifacts.

    Args:
        session_dir: The directory for the current session
        snapshot: Snapshot information, if available
        save_main_script: Whether to save the main script/notebook
        save_git_diff: Whether to save the git diff

    Returns:
        A tuple of (saved_script_path, script_type) if main script was saved, (None, None) otherwise
    """
    log.debug(f"Setting up code directory in {session_dir}")

    # Always create the code directory
    code_dir = resolve_code_directory(session_dir)

    # Create copy of snapshots if available
    if snapshot is not None:
        # By default, our snapshot directory is already inside the code directory,
        # so we only need to copy the snapshot directory if it's not the same as the code directory.
        if snapshot.snapshot_dir != code_dir:
            log.debug(f"Creating snapshot copy in {code_dir}")
            _create_snapshot_copy(snapshot, code_dir)
        else:
            log.debug(
                "Snapshot directory is the same as code directory, no copy needed"
            )

    # Save git diff if enabled
    git_diff_path: Path | None = None
    if save_git_diff:
        git_diff_path = _save_git_diff(code_dir)
    else:
        log.debug("Git diff saving is disabled")

    # Save main script if enabled
    saved_script_path: Path | None = None
    script_type: Literal["script", "notebook"] | None = None
    if save_main_script:
        saved_script_path, script_type = _save_main_script_file(code_dir)
        if saved_script_path:
            log.debug(f"Main script saved to {saved_script_path}")
        else:
            log.debug("Failed to save main script")
    else:
        log.debug("Main script saving is disabled")

    return CodeDirectoryResult(
        code_dir=code_dir,
        saved_script_path=saved_script_path,
        script_type=script_type,
        git_diff_path=git_diff_path,
    )
