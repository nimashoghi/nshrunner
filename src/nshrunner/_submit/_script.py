from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path


def _create_launcher_script_file(
    script_path: Path,
    original_command: str | Iterable[str],
    environment: Mapping[str, str],
    setup_commands: Sequence[str],
    chmod: bool = True,
    prepend_command_with_exec: bool = True,
    command_prefix: str | None = None,
    # ^ If True, the original command will be prepended with 'exec' to replace the shell process
    #   with the command. This is useful for ensuring that the command is the only process in the
    #   process tree (e.g. for better signal handling).
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

        if not isinstance(original_command, str):
            original_command = " ".join(original_command)

        if command_prefix:
            original_command = f"{command_prefix} {original_command}"

        if prepend_command_with_exec:
            original_command = f"exec {original_command}"
        f.write(f"{original_command}\n")

    if chmod:
        # Make the script executable
        script_path.chmod(0o755)


def write_helper_script(
    base_dir: Path,
    command: str | Iterable[str],
    environment: Mapping[str, str],
    setup_commands: Sequence[str],
    chmod: bool = True,
    prepend_command_with_exec: bool = True,
    command_prefix: str | None = None,
    file_name: str = "helper.sh",
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

    out_path = base_dir / file_name
    _create_launcher_script_file(
        out_path,
        command,
        environment,
        setup_commands,
        chmod,
        prepend_command_with_exec,
        command_prefix,
    )
    return out_path


DEFAULT_TEMPLATE = "bash {script}"


def helper_script_to_command(script: Path, template: str | None) -> str:
    if not template:
        template = DEFAULT_TEMPLATE

    # Make sure the template has '{script}' in it
    if "{script}" not in template:
        raise ValueError(f"Template must contain '{{script}}'. Got: {template!r}")

    return template.format(script=str(script.absolute()))
