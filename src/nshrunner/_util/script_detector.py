from __future__ import annotations

import inspect
import logging
import shutil
import sys
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)


def get_executed_notebook_code() -> str | None:
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


def get_main_script() -> tuple[Path | None, Literal["script", "notebook"] | None]:
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


def save_main_script(
    destination_dir: Path,
) -> tuple[Path | None, Literal["script", "notebook"] | None]:
    """
    Save the main executing script or notebook to the specified directory.

    Args:
        destination_dir: The directory where the script should be saved

    Returns:
        A tuple of (saved_file_path, file_type) if successful, (None, None) otherwise.
    """
    log.debug(f"Attempting to save main script to: {destination_dir}")
    script_path, script_type = get_main_script()

    # Create the destination directory if it doesn't exist
    destination_dir.mkdir(parents=True, exist_ok=True)

    # Special handling for notebooks - use IPython history to create a Python script
    if script_type == "notebook" and script_path is None:
        log.debug("Creating notebook script from IPython history")
        notebook_code = get_executed_notebook_code()

        if notebook_code:
            try:
                destination_file = destination_dir / "main_notebook.py"
                with open(destination_file, "w") as f:
                    f.write(notebook_code)
                log.debug(
                    f"Successfully saved executed notebook code to {destination_file}"
                )
                return destination_file, "notebook"
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
        destination_file = destination_dir / f"main_{script_type}{script_path.suffix}"
        log.debug(f"Copying {script_path} to {destination_file}")
        shutil.copy2(script_path, destination_file)
        log.debug(f"Successfully saved {script_type} to {destination_file}")
        return destination_file, script_type
    except Exception as e:
        log.debug(f"Error saving {script_type}: {e}")
        return None, None
