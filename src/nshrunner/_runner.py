from __future__ import annotations

import functools
import logging
import sys
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, cast

import nshsnap
from typing_extensions import TypeVar, TypeVarTuple, Unpack

from . import _env
from ._config import Config
from ._seed import seed_everything
from ._submit import screen, slurm
from ._util.env import with_env
from ._util.environment import (
    remove_nshrunner_environment_variables,
    remove_slurm_environment_variables,
    remove_wandb_environment_variables,
)
from ._util.git import gitignored_dir

log = logging.getLogger(__name__)


TArguments = TypeVarTuple("TArguments")
TReturn = TypeVar("TReturn", infer_variance=True)


@dataclass
class _Session:
    id: str
    """The ID of the session."""

    dir_path: Path
    """The path to the session directory."""

    env: dict[str, str] = field(default_factory=lambda: {})
    """Environment variables to set for the session."""

    snapshot: nshsnap.SnapshotInfo | None = None
    """The snapshot information for the session."""


T = TypeVar("T", infer_variance=True)


def _tqdm_if_installed(iterable: Iterable[T], *args, **kwargs) -> Iterable[T]:
    try:
        from tqdm.auto import tqdm  # pyright: ignore[reportMissingImports]

        return cast(Iterable[T], tqdm(iterable, *args, **kwargs))
    except ImportError:
        return iterable


def _wrap_run_fn(config: Config, run_fn: Callable[[Unpack[TArguments]], TReturn]):
    @functools.wraps(run_fn)
    def wrapped_run_fn(*args: Unpack[TArguments]) -> TReturn:
        # Seed
        if (seed := config._resolve_seed_config()) is not None:
            seed_everything(seed)

        return run_fn(*args)

    return wrapped_run_fn


def _shell_hook(env_path: Path):
    # Detect the environment type
    if env_path.joinpath("conda-meta", "history").exists():
        # Conda/Mamba environment
        return f'eval "$(conda shell.bash hook)" && conda activate "{env_path}"'
    elif env_path.joinpath("bin", "activate").exists():
        # Venv or Poetry environment
        return f'source "{env_path}/bin/activate"'
    else:
        raise ValueError(f"Unable to detect the environment type for {env_path}")


class Runner(Generic[TReturn, Unpack[TArguments]]):
    def generate_id(self):
        return str(uuid.uuid4())

    def __init__(
        self,
        run_fn: Callable[[Unpack[TArguments]], TReturn],
        config: Config = Config(),
    ):
        self.config = config
        self.run_fn = run_fn

    @property
    def _wrapped_run_fn(self):
        return _wrap_run_fn(self.config, self.run_fn)

    def _setup_session(
        self,
        runs: Iterable[tuple[Unpack[TArguments]]],
        id: str | None = None,
        *,
        env: Mapping[str, str] | None,
    ):
        runs = list(runs)

        # Create id if not provided
        if id is None:
            id = self.generate_id()

        # Create the session directory
        working_dir = gitignored_dir(self.config._resolve_working_dir())
        session_dir = gitignored_dir(working_dir / id, create=True)

        # Create the session object (to return)
        session = _Session(id=id, dir_path=session_dir)

        # Resolve the environment
        session.env = {
            _env.SESSION_ID: id,
            _env.SESSION_DIR: str(session.dir_path.resolve().absolute()),
            **session.env,
            **(self.config.env or {}),
            **(env or {}),
        }

        # Take a snapshot of the environment if needed
        if (snapshot := self.config._resolve_snapshot_config(session_dir)) is not None:
            session.snapshot = nshsnap.snapshot(snapshot)
            snapshot_path_str = str(session.snapshot.snapshot_dir.absolute())

            # Update the environment to include the snapshot path and
            # prepend the new PYTHONPATH to the env dict.
            session.env = {
                "PYTHONPATH": f"{snapshot_path_str}:$PYTHONPATH",
                **session.env,
                _env.SNAPSHOT_DIR: snapshot_path_str,
                _env.SNAPSHOT_MODULES: ",".join(session.snapshot.modules),
            }

        # Create code directory and save main script/git diff as configured
        from ._util.code_saving import setup_code_directory

        result = setup_code_directory(
            session_dir,
            session.snapshot,
            save_main_script=self.config.save_main_script,
            save_git_diff=self.config.save_git_diff,
        )

        # Add information to environment
        session.env[_env.CODE_DIR] = str(result.code_dir.resolve().absolute())
        if result.saved_script_path is not None:
            session.env[_env.MAIN_SCRIPT_PATH] = str(
                result.saved_script_path.resolve().absolute()
            )
        if result.script_type is not None:
            session.env[_env.MAIN_SCRIPT_TYPE] = result.script_type
        if result.git_diff_path is not None:
            session.env[_env.GIT_DIFF_PATH] = str(
                result.git_diff_path.resolve().absolute()
            )

        return runs, session

    def local_generator(
        self,
        runs: Iterable[tuple[Unpack[TArguments]]],
        *,
        env: Mapping[str, str] | None = None,
    ):
        runs, session = self._setup_session(runs, env=env)
        with with_env(session.env):
            for args in _tqdm_if_installed(runs):
                yield self._wrapped_run_fn(*args)

    def local(
        self,
        runs: Iterable[tuple[Unpack[TArguments]]],
        *,
        env: Mapping[str, str] | None = None,
    ):
        return list(self.local_generator(runs, env=env))

    @remove_nshrunner_environment_variables()
    @remove_slurm_environment_variables()
    @remove_wandb_environment_variables()
    def session(
        self,
        runs: Iterable[tuple[Unpack[TArguments]]],
        options: screen.ScreenJobKwargs = {},
        *,
        setup_commands: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        activate_venv: bool = True,
        print_command: bool = True,
    ):
        # Resolve all runs
        runs, session = self._setup_session(runs, env=env)
        base_dir = session.dir_path / "submit"
        base_dir.mkdir(parents=True, exist_ok=True)

        # Update the job options
        options = screen.update_options(options, base_dir)

        # Use setup commands to directly put env/pythonpath into the session bash script
        setup_commands_pre: list[str] = []
        if activate_venv:
            setup_commands_pre.append("echo 'Activating environment'")
            setup_commands_pre.append(_shell_hook(Path(sys.prefix)))

        # Merge the setup commands
        setup_commands = (
            setup_commands_pre
            + list(setup_commands or [])
            + list(options.get("setup_commands", []))
        )
        del setup_commands_pre

        # Merge the environment
        env = {**session.env, **options.get("environment", {})}

        # Convert runs to commands using picklerunner
        from .picklerunner.create import callable_to_command

        command = callable_to_command(
            base_dir / "worker.sh",
            self._wrapped_run_fn,
            runs,
            environment=env,
            setup_commands=setup_commands,
            execution={
                "mode": "sequential",
                "pause_before_exit": options.get("pause_before_exit", True),
            },
        )

        # Create the submission script
        submission = screen.to_array_batch_script(
            command,
            script_path=base_dir / "submit.sh",
            config=options,
            env=env,
        )

        # Print the full command so the user can copy-paste it
        if print_command:
            log.critical("Run the following command to submit the jobs:\n\n")
            # We print the command but log the rest so the user can pipe the command to bash
            print(f"{submission.command_str}\n\n")

        return submission

    @remove_nshrunner_environment_variables()
    @remove_slurm_environment_variables()
    @remove_wandb_environment_variables()
    def submit_slurm(
        self,
        runs: Iterable[tuple[Unpack[TArguments]]],
        options: slurm.SlurmJobKwargs,
        *,
        setup_commands: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        activate_venv: bool = True,
        print_command: bool = True,
    ):
        # Resolve all runs
        runs, session = self._setup_session(runs, env=env)
        base_dir = session.dir_path / "submit"
        base_dir.mkdir(parents=True, exist_ok=True)

        # Update the SLURM options
        options = slurm.update_options(options, base_dir)

        # Use setup commands to directly put env/pythonpath into the session bash script
        setup_commands_pre: list[str] = []
        if activate_venv:
            setup_commands_pre.append("echo 'Activating environment'")
            setup_commands_pre.append(_shell_hook(Path(sys.prefix)))

        # Merge the setup commands
        setup_commands = (
            setup_commands_pre
            + list(setup_commands or [])
            + list(options.get("setup_commands", []))
        )
        del setup_commands_pre

        # Merge the environment
        env = {**session.env, **options.get("environment", {})}

        # Convert runs to commands using picklerunner
        from .picklerunner.create import callable_to_command

        command = callable_to_command(
            base_dir / "worker.sh",
            self._wrapped_run_fn,
            runs,
            environment=env,
            setup_commands=setup_commands,
            execution={"mode": "array"},
        )

        # Create the submission script
        submission = slurm.to_array_batch_script(
            command,
            script_path=base_dir / "submit.sh",
            num_jobs=len(runs),
            config=options,
            env=env,
        )

        # Print the full command so the user can copy-paste it
        if print_command:
            log.critical("Run the following command to submit the jobs:\n\n")
            # We print the command but log the rest so the user can pipe the command to bash
            print(f"{submission.command_str}\n\n")

        return submission
