# %%
from __future__ import annotations

import nshrunner as R


def run_fn(x: int):
    return x + 5


runs = [(1,)]

runner = R.Runner(run_fn, R.RunnerConfig(working_dir="."))

# %%
list(runner.local(runs))

# %%
runner.session(
    runs,
    {},
    snapshot=True,
    env={"CUDA_VISIBLE_DEVICES": "0"},
)

# %%
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
