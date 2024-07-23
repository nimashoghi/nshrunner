from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path


def write_helper_script(
    script_path: Path,
    command: str | Iterable[str],
    environment: Mapping[str, str],
    setup_commands: Sequence[str],
    chmod: bool = True,
    prepend_command_with_exec: bool = True,
    command_prefix: str | None = None,
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

        if not isinstance(command, str):
            command = " ".join(command)

        if command_prefix:
            command = f"{command_prefix} {command}"

        if prepend_command_with_exec:
            command = f"exec {command}"
        f.write(f"{command}\n")

    if chmod:
        # Make the script executable
        script_path.chmod(0o755)


DEFAULT_TEMPLATE = "bash {script}"


def helper_script_to_command(script: Path, template: str | None) -> str:
    if not template:
        template = DEFAULT_TEMPLATE

    # Make sure the template has '{script}' in it
    if "{script}" not in template:
        raise ValueError(f"Template must contain '{{script}}'. Got: {template!r}")

    return template.format(script=str(script.absolute()))
