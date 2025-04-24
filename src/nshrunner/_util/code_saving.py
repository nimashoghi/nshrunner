from __future__ import annotations

import inspect
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from nshsnap import SnapshotInfo

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


def _is_git_repo(path: Path) -> bool:
    """
    Check if the given path is within a Git repository.

    Args:
        path: The path to check

    Returns:
        True if the path is within a Git repository, False otherwise
    """
    try:
        # Try to get the Git root directory
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def _get_git_root(path: Path) -> Path | None:
    """
    Get the root directory of the Git repository containing the given path.

    Args:
        path: A path within the Git repository

    Returns:
        The root directory of the Git repository, or None if not in a repository
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
        return None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def _is_untracked_file(repo_path: Path, file_path: Path) -> bool:
    """
    Check if a file is untracked in the Git repository.

    Args:
        repo_path: The path to the Git repository
        file_path: The path to the file to check

    Returns:
        True if the file is untracked, False otherwise
    """
    try:
        # Use git ls-files to check if the file is tracked
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_path),
                "ls-files",
                "--error-unmatch",
                str(file_path.relative_to(repo_path)),
            ],
            capture_output=True,
            check=False,
        )
        # Return code 0 means the file is tracked, 1 means it's untracked
        return result.returncode != 0
    except (subprocess.SubprocessError, ValueError):
        # If there's an error or the file is not relative to the repo, consider it untracked
        return True


def _get_git_diff(repo_path: Path) -> str | None:
    """
    Get the current Git diff for the repository.

    Args:
        repo_path: The path to the Git repository

    Returns:
        The Git diff as a string, or None if there was an error
    """
    try:
        # Get the diff for tracked files
        tracked_diff = subprocess.run(
            ["git", "-C", str(repo_path), "diff", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )

        # Get the diff for untracked files
        untracked_files = []
        for f in repo_path.glob("**/*"):
            if f.is_file() and _is_untracked_file(repo_path, f):
                untracked_files.append(str(f))

        untracked_diff_output = ""
        if untracked_files:
            try:
                # Get the diff for untracked files
                untracked_diff = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(repo_path),
                        "diff",
                        "--no-index",
                        "--",
                        "/dev/null",
                    ]
                    + untracked_files,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                # The --no-index command returns 1 if there are differences, which is expected
                untracked_diff_output = (
                    untracked_diff.stdout if untracked_diff.returncode in (0, 1) else ""
                )
            except (subprocess.SubprocessError, FileNotFoundError) as e:
                log.debug(f"Error getting diff for untracked files: {e}")

        # Combine the diffs
        combined_diff = tracked_diff.stdout + "\n" + untracked_diff_output

        return combined_diff if combined_diff.strip() else None
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        log.debug(f"Error getting Git diff: {e}")
        return None


def _create_code_directory(session_dir: Path) -> Path:
    """
    Create a code directory structure for the run.

    Args:
        session_dir: The directory for the current session

    Returns:
        The path to the created code directory
    """
    # Create the code directory
    code_dir = session_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)

    log.debug(f"Created code directory at {code_dir}")
    return code_dir


def _create_snapshot_symlink(snapshot: SnapshotInfo, code_dir: Path) -> Path | None:
    """
    Create symlink to dir that contains snapshot modules.

    Args:
        snapshot: Snapshot information containing the directory
        code_dir: The code directory where symlink should be created

    Returns:
        The path to the created symlink directory if successful, None otherwise
    """
    if not (snapshot_dir_path := Path(snapshot.snapshot_dir)).exists():
        log.debug(f"Snapshot directory {snapshot_dir_path} does not exist")
        return None

    snapshot_symlink_path = (code_dir / "snapshot").resolve().absolute()
    # Check if a symlink already exists at the destination and remove it if necessary
    if snapshot_symlink_path.exists() or snapshot_symlink_path.is_symlink():
        try:
            log.debug(f"Removing existing symlink or file at {snapshot_symlink_path}")
            snapshot_symlink_path.unlink(missing_ok=True)
        except Exception as e:
            log.debug(f"Error removing existing symlink: {e}")
            return None

    try:
        log.debug(
            f"Creating symlink from {snapshot_symlink_path} to {snapshot_dir_path}"
        )
        # Use absolute paths for both source and destination
        snapshot_symlink_path.symlink_to(
            snapshot_dir_path.resolve().absolute(), target_is_directory=True
        )
        log.debug(f"Created symlink at {snapshot_symlink_path}")
        return snapshot_symlink_path
    except Exception as e:
        log.debug(f"Error creating symlink to snapshot directory: {e}")
        return None


def _save_git_diff(code_dir: Path) -> Path | None:
    """
    Save the git diff to the code directory if in a git repository.

    Args:
        code_dir: The code directory where the git diff should be saved

    Returns:
        Path to the saved git diff file if successful, None otherwise
    """
    # Check if we're in a Git repository and save the diff
    cwd = Path.cwd()
    if not _is_git_repo(cwd):
        log.debug("Not in a Git repository")
        return None

    repo_root = _get_git_root(cwd)
    if not repo_root:
        log.debug("Failed to determine Git repository root")
        return None

    log.debug(f"Detected Git repository at {repo_root}")

    # Get the Git diff
    diff = _get_git_diff(repo_root)

    # Create diff file path
    diff_file = code_dir / "git_diff.patch"

    # Always create the file for valid repositories, even if there are no changes
    if diff:
        log.debug("Repository has changes, saving git diff")
        with open(diff_file, "w") as f:
            f.write(diff)
        log.debug(f"Saved Git diff to {diff_file}")
    else:
        log.debug("Repository is clean, creating empty git diff file")
        # Create an empty file
        diff_file.touch()
        log.debug(f"Created empty git diff file at {diff_file}")

    return diff_file


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
    code_dir = _create_code_directory(session_dir)

    # Create symlink to snapshots if available
    if snapshot is not None:
        _create_snapshot_symlink(snapshot, code_dir)

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
