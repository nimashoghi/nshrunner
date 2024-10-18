# nshrunner

nshrunner is a Python library that provides a unified way to run functions in various environments, such as local dev machines, cloud VMs, SLURM clusters, and LSF clusters. It was created to simplify the process of running ML training jobs across multiple machines and environments.

## Motivation

When running ML training jobs on different machines and environments, it can be challenging to manage the specifics of each environment. nshrunner was developed to address this issue by providing a single function that can be used to run jobs on any supported environment without having to worry about the details of each environment.

## Features

- Supports running functions locally, on SLURM clusters, and on LSF clusters
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

Here's a simple example of how to use nshrunner to run a function locally:

```python
import nshrunner as R

def run_fn(x: int):
    return x + 5

runs = [(1,)]

runner = R.Runner(run_fn, R.RunnerConfig(working_dir="."))
list(runner.local(runs))
```

To run the same function on a SLURM cluster:

```python
runner.submit_slurm(
    runs,
    {
        "partition": "learnaccel",
        "nodes": 4,
        "ntasks_per_node": 8,  # Change this to limit # of GPUs
        "gpus_per_task": 1,
        "cpus_per_task": 1,
    },
    snapshot=True,
)
```

And on an LSF cluster:

```python
runner.submit_lsf(
    runs,
    {
        "summit": True,
        "queue": "learnaccel",
        "nodes": 4,
        "rs_per_node": 8,  # Change this to limit # of GPUs
    },
    snapshot=True,
)
```

For more detailed usage examples, please refer to the documentation.

## Acknowledgements

`nshrunner` is heavily inspired by [`submitit`](https://github.com/facebookincubator/submitit). It builds on `submitit`'s design and adds support for LSF clusters, snapshotting, and other features.

## Contributing

Contributions are welcome! For feature requests, bug reports, or questions, please open an issue on GitHub. If you'd like to contribute code, please submit a pull request with your changes.

## License

nshrunner is released under the MIT License. See `LICENSE` for more information.
