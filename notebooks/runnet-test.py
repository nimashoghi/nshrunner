# %%
import nshrunner


def run_fn(x: int):
    return x + 5


runs = [(1,)]

config = nshrunner.Config()
runner = nshrunner.Runner(config, run_fn)

# %%
list(runner.local(runs))

# %%
runner.session(runs, snapshot=False, pause_before_exit=True)
