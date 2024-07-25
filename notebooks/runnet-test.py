# %%
import nshrunner as R


def run_fn(x: int):
    return x + 5


runs = [(1,)]

runner = R.Runner(run_fn)

# %%
list(runner.local(runs))

# %%
runner.session(runs, snapshot=False, pause_before_exit=True)

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

# %%

runner.submit_lsf(
    runs,
    {
        "queue": "learnaccel",
        "nodes": 4,
        "rs_per_node": 8,  # Change this to limit # of GPUs
    },
    snapshot=True,
)

# %%
