from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Literal

from typing_extensions import Unpack

from ._runner import Runner, TArguments, TReturn
from ._submit._util import Submission

if TYPE_CHECKING:
    from . import configs


def run_local(
    fn: Callable[[Unpack[TArguments]], TReturn],
    args_list: Iterable[tuple[Unpack[TArguments]]],
    *,
    runner: configs.ConfigInstanceOrDict = {},
):
    """Run function locally with multiple argument sets.

    Parameters
    ----------
    fn : Callable[[Unpack[TArguments]], TReturn]
        Function to execute
    args_list : Iterable[tuple[Unpack[TArguments]]]
        Iterable of argument tuples to pass to the function
    runner : configs.ConfigInstanceOrDict, optional
        Configuration for the runner (e.g. working directory, snapshot, etc.)

    Returns
    -------
    TReturn
        The return value(s) from the function execution
    """
    from . import configs

    runner_config = configs.CreateConfig(runner)
    assert not runner_config.snapshot, "Snapshot is not supported for local runs"
    return Runner(fn, runner_config).local(args_list)


def submit_screen(
    fn: Callable[[Unpack[TArguments]], TReturn],
    args_list: Iterable[tuple[Unpack[TArguments]]],
    *,
    screen: configs.ScreenBackendConfigInstanceOrDict = {},
    runner: configs.ConfigInstanceOrDict = {},
):
    """Submit function using GNU Screen with multiple argument sets.

    Parameters
    ----------
    fn : Callable[[Unpack[TArguments]], TReturn]
        Function to execute
    args_list : Iterable[tuple[Unpack[TArguments]]]
        Iterable of argument tuples to pass to the function
    screen : configs.ScreenBackendConfigInstanceOrDict
        Configuration for the Screen backend
    runner : configs.ConfigInstanceOrDict, optional
        Configuration for the runner (e.g. working directory, snapshot, etc.)

    Returns
    -------
    Submission
        The submission object containing command and script information
    """
    from . import configs

    runner_config = configs.CreateConfig(runner)
    screen_config = configs.CreateScreenBackendConfig(screen)
    return Runner(fn, runner_config).session(
        args_list, screen_config.to_screen_kwargs()
    )


def submit_parallel_screen(
    fn: Callable[[Unpack[TArguments]], TReturn],
    args_list: Iterable[tuple[Unpack[TArguments]]],
    *,
    num_parallel_screens: int,
    screen: configs.ScreenBackendConfigInstanceOrDict = {},
    runner: configs.ConfigInstanceOrDict = {},
    scheduling: Literal["round_robin", "block"] = "round_robin",
    runner_config_modifier: Callable[[int, configs.Config], configs.Config | None]
    | None = None,
    screen_config_modifier: Callable[
        [int, configs.ScreenBackendConfig], configs.ScreenBackendConfig | None
    ]
    | None = None,
):
    """Submit function using multiple GNU Screen sessions with per-session configuration overrides.

    Parameters
    ----------
    fn : Callable[[Unpack[TArguments]], TReturn]
        Function to execute
    args_list : Iterable[tuple[Unpack[TArguments]]]
        Iterable of argument tuples to pass to the function
    num_parallel_screens : int
        Number of parallel screen sessions to create
    screen : configs.ScreenBackendConfigInstanceOrDict
        Configuration for the Screen backend
    runner : configs.ConfigInstanceOrDict, optional
        Configuration for the runner (e.g. working directory, snapshot, etc.)
    scheduling : {"round_robin", "block"}, optional
        Scheduling mode for distributing argument sets. "round_robin" (default) assigns arguments
        in an alternating sequence (e.g. [(0,), (2,)] and [(1,), (3,)] for 2 screens). "block"
        divides the list into contiguous chunks (e.g. [(0,), (1,)] and [(2,), (3,)]).
    runner_config_modifier : callable, optional
        A callable that takes the session index and the runner config, and applies modifications.
    screen_config_modifier : callable, optional
        A callable that takes the session index and the screen config, and applies modifications.

    Returns
    -------
    list[Submission]
        List of submission objects containing command and script information for each screen session
    """
    from . import configs

    # Convert args_list to list for indexing
    args_list = list(args_list)

    # Validate num_parallel_screens
    if num_parallel_screens <= 0:
        raise ValueError("num_parallel_screens must be positive")
    if num_parallel_screens > len(args_list):
        num_parallel_screens = len(args_list)

    runner_config = configs.CreateConfig(runner)
    screen_config = configs.CreateScreenBackendConfig(screen)

    # Create separate screen sessions for each partition
    submissions: list[Submission] = []
    for i in range(num_parallel_screens):
        runner_config_screen = runner_config.model_copy(deep=True)
        screen_config_screen = screen_config.model_copy(deep=True)

        if runner_config_modifier:
            if (ret := runner_config_modifier(i, runner_config_screen)) is not None:
                runner_config_screen = ret

        if screen_config_modifier:
            if (ret := screen_config_modifier(i, screen_config_screen)) is not None:
                screen_config_screen = ret

        if scheduling == "round_robin":
            partition = args_list[i::num_parallel_screens]
        elif scheduling == "block":
            chunk_size = (
                len(args_list) + num_parallel_screens - 1
            ) // num_parallel_screens
            partition = args_list[i * chunk_size : (i + 1) * chunk_size]
        else:
            raise ValueError(f"Invalid scheduling mode: {scheduling}")
        submission = Runner(fn, runner_config_screen).session(
            partition, screen_config_screen.to_screen_kwargs()
        )
        submissions.append(submission)

    return submissions


def submit_slurm(
    fn: Callable[[Unpack[TArguments]], TReturn],
    args_list: Iterable[tuple[Unpack[TArguments]]],
    *,
    slurm: configs.SlurmBackendConfigInstanceOrDict,
    runner: configs.ConfigInstanceOrDict = {},
):
    """Submit function to Slurm with multiple argument sets.

    Parameters
    ----------
    fn : Callable[[Unpack[TArguments]], TReturn]
        Function to execute
    args_list : Iterable[tuple[Unpack[TArguments]]]
        Iterable of argument tuples to pass to the function
    slurm : configs.SlurmBackendConfigInstanceOrDict
        Configuration for the Slurm backend
    runner : configs.ConfigInstanceOrDict, optional
        Configuration for the runner (e.g. working directory, snapshot, etc.)

    Returns
    -------
    Submission
        The submission object
    """
    from . import configs

    runner_config = configs.CreateConfig(runner)
    slurm_config = configs.CreateSlurmBackendConfig(slurm)
    return Runner(fn, runner_config).submit_slurm(
        args_list, slurm_config.to_slurm_kwargs()
    )
