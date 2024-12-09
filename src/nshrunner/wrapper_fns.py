from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

from typing_extensions import Unpack

from ._runner import Config, Runner, Snapshot, TArguments, TReturn
from ._submit import slurm

try:
    from .configs.Config_typed_dict import ConfigTypedDict as ConfigTypedDict
except:
    if TYPE_CHECKING:
        raise ValueError("ConfigTypedDict not found")


def run_local(
    fn: Callable[[Unpack[TArguments]], TReturn],
    args_list: Iterable[tuple[Unpack[TArguments]]],
    **kwargs: Unpack[ConfigTypedDict],
):
    env = kwargs.pop("env", None)
    runner = Runner(fn, Config.from_dict({**kwargs}))
    return runner.local(args_list, env=env)


def submit_slurm(
    fn: Callable[[Unpack[TArguments]], TReturn],
    args_list: Iterable[tuple[Unpack[TArguments]]],
    options: slurm.SlurmJobKwargs,
    config: Config = Config(),
    *,
    snapshot: Snapshot,
    setup_commands: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    activate_venv: bool = True,
):
    runner = Runner(fn, config)
    return runner.submit_slurm(
        args_list,
        options,
        snapshot=snapshot,
        setup_commands=setup_commands,
        env=env,
        activate_venv=activate_venv,
    )
