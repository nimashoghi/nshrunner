from __future__ import annotations

import functools
import logging
import sys
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, Literal, cast

import nshconfig as C
import nshsnap
from typing_extensions import TypeVar, TypeVarTuple, Unpack

from . import _env
from ._seed import SeedConfig, seed_everything
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


class Config(C.Config):
    working_dir: str | Path | Literal["cwd", "tmp", "home-cache"] = "home-cache"
    """
    The `working_dir` parameter is a string that represents the directory where the program will save its execution files and logs.
    This is used when submitting the program to a SLURM/LSF cluster or when using the `local_sessions` method.

    Accepted values are:
    - "cwd": The current working directory.
    - "tmp": The temporary directory.
    - "home-cache" (default): The cache directory in the user's home directory (i.e., `~/.cache/nshrunner`).

    """

    seed: int | SeedConfig | None = SeedConfig(seed=0)
    """Seed configuration for the runner."""

    env: Mapping[str, str] | None = None
    """Environment variables to set for the session."""

    snapshot: bool | nshsnap.SnapshotConfig = False
    """Snapshot configuration for the session."""

    def _resolve_seed_config(self):
        if self.seed is None:
            return None

        if isinstance(self.seed, int):
            return SeedConfig(seed=self.seed)

        return self.seed

    def _resolve_working_dir(self):
        match self.working_dir:
            case "cwd":
                return Path.cwd() / "nshrunner"
            case "tmp":
                return Path("/tmp") / "nshrunner"
            case "home-cache":
                return Path.home() / ".cache" / "nshrunner"
            case _:
                return Path(self.working_dir)


T = TypeVar("T", infer_variance=True)


def _tqdm_if_installed(iterable: Iterable[T], *args, **kwargs) -> Iterable[T]:
    try:
        from tqdm.auto import tqdm  # type: ignore

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
        if snapshot := self.config.snapshot:
            # If the snapshot is not a SnapshotConfig object, create one
            if not isinstance(snapshot, nshsnap.SnapshotConfig):
                if snapshot is True:
                    snapshot = {}

                # Merge the default snapshot kwargs
                snapshot = {**snapshot}

                # If the save directory is not set, set it to the session directory
                if not snapshot.get("snapshot_dir"):
                    snapshot_dir = gitignored_dir(session_dir / "nshsnap", create=True)
                    snapshot["snapshot_dir"] = snapshot_dir

                snapshot = nshsnap.configs.SnapshotConfig.from_dict(snapshot)

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
