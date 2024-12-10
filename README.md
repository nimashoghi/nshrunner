# nshrunner

nshrunner is a Python library that provides a unified way to run functions in various environments, such as local dev machines, cloud VMs, and SLURM clusters. It was created to simplify the process of running ML training jobs across multiple machines and environments.

## Motivation

When running ML training jobs on different machines and environments, it can be challenging to manage the specifics of each environment. nshrunner was developed to address this issue by providing a single function that can be used to run jobs on any supported environment without having to worry about the details of each environment.

## Features

- Supports running functions locally, on SLURM clusters, and in GNU Screen sessions
- Provides a unified interface for running functions across different environments
- Allows for easy configuration of job options, such as resource requirements and environment variables
- Supports snapshotting the environment to ensure reproducibility, using the [`nshsnap`](https://www.github.com/nimashoghi/nshsnap) library
- Provides utilities for logging, seeding, and signal handling

## Installation

nshrunner can be installed using pip:

```bash
pip install nshrunner
```

## Usage

Here's a simple example showing the different ways to run a function:

```python
import nshrunner as R

def train_model(batch_size: int, learning_rate: float):
    # Training logic here
    return {"accuracy": 0.95}

# Define runs with different hyperparameters
runs = [
    (32, 0.001),  # (batch_size, learning_rate)
    (64, 0.0005),
]

# Run locally
results = R.run_local(train_model, runs)

# Run in a GNU Screen session
R.submit_screen(
    train_model,
    runs,
    screen={
        "name": "training",
        "logging": {
            "output_file": "logs/output.log",
            "error_file": "logs/error.log"
        },
        "attach": False  # Run detached
    }
)

# Run on SLURM
R.submit_slurm(
    train_model,
    runs,
    slurm={
        "name": "training",
        "partition": "gpu",
        "resources": {
            "nodes": 1,
            "cpus": 4,
            "gpus": 1,
            "memory_gb": 32,
            "time": "12:00:00"
        },
        "output_dir": "logs"
    }
)
```

The library provides a consistent interface across different execution environments while handling the complexities of:

- Job submission and management
- Resource allocation
- Environment setup
- Output logging
- Error handling

For more advanced usage, you can configure additional options like:

```python
# Configure environment snapshot for reproducibility
R.submit_slurm(
    train_model,
    runs,
    runner={
        "working_dir": "experiments",
        "snapshot": True,  # Snapshot code and dependencies
        "seed": {"seed": 42}  # Set random seeds
    },
    slurm={...}
)
```

## Contributing

Contributions are welcome! For feature requests, bug reports, or questions, please open an issue on GitHub. If you'd like to contribute code, please submit a pull request with your changes.

## License

nshrunner is released under the MIT License. See `LICENSE` for more information.
