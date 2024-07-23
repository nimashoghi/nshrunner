SHELL_HOOK = r"""\
#!/bin/bash

activate_environment() {
    local env_path="$1"

    if [[ ! -d "$env_path" ]]; then
        echo "Error: The provided path does not exist or is not a directory."
        return 1
    fi

    # Check for conda/mamba environment
    if [[ -f "$env_path/conda-meta/history" ]]; then
        echo "Detected conda/mamba environment"
        if command -v conda &> /dev/null; then
            conda activate "$env_path"
        elif command -v mamba &> /dev/null; then
            mamba activate "$env_path"
        else
            echo "Error: conda/mamba is not available in the current shell"
            return 1
        fi
    # Check for venv
    elif [[ -f "$env_path/bin/activate" ]]; then
        echo "Detected venv environment"
        source "$env_path/bin/activate"
    # Check for poetry
    elif [[ -f "$env_path/poetry.lock" ]]; then
        echo "Detected poetry environment"
        if command -v poetry &> /dev/null; then
            cd "$env_path"
            poetry shell
        else
            echo "Error: poetry is not available in the current shell"
            return 1
        fi
    else
        echo "Error: Unable to detect the environment type"
        return 1
    fi

    echo "Environment activated successfully"
}

# Check if a path argument is provided
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <path_to_environment>"
    exit 1
fi

activate_environment "$1"
"""
