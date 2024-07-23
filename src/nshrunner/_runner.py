import contextlib
import copy
import functools
import logging
import os
import shutil
import subprocess
import sys
import traceback
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property, wraps
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Generic,
    Literal,
    Protocol,
    TypeAlias,
    cast,
    runtime_checkable,
)

import nshconfig as C
from typing_extensions import (
    ParamSpec,
    TypedDict,
    TypeVar,
    TypeVarTuple,
    Unpack,
    override,
)

from ._logging import PythonLoggingConfig, init_python_logging
from ._seed import SeedConfig, seed_everything
from ._submit import unified
from ._submit._script import write_helper_script
from ._util.env import _with_env
from ._util.environment import (
    remove_lsf_environment_variables,
    remove_slurm_environment_variables,
    remove_wandb_environment_variables,
)
from ._util.git import _gitignored_dir
from .model.config import BaseConfig
from .snapshot import SnapshotArgType, SnapshotConfig, snapshot_modules
from .trainer import Trainer

log = logging.getLogger(__name__)


T = TypeVar("T", infer_variance=True)


_Path: TypeAlias = str | Path | os.PathLike


TArguments = TypeVarTuple("TArguments")
TReturn = TypeVar("TReturn", infer_variance=True)


class RunInfo(TypedDict, total=False):
    id: str
    """The ID of the run."""

    base_dir: _Path
    """The base directory to save the run's files to."""

    env: Mapping[str, str]
    """Environment variables to set for the run."""

    skip_python_logging: bool
    """Whether to skip setting up Python logging for the run. Default: `False`."""


class Config(C.Config):
    savedir: _Path | None = None
    """
    The `savedir` parameter is a string that represents the directory where the program will save its execution files and logs.
        This is used when submitting the program to a SLURM/LSF cluster or when using the `local_sessions` method.
        If `None`, this will default to the current working directory / `llrunner`.
    """

    python_logging: PythonLoggingConfig = PythonLoggingConfig()
    """Logging configuration for the runner."""

    seed: SeedConfig = SeedConfig(seed=0)
    """Seed configuration for the runner."""

    env: Mapping[str, str] | None = None
    """Environment variables to set for the session."""

    validate_no_duplicate_ids: bool = True
    """Whether to validate that there are no duplicate IDs in the runs."""

    snapshot_env_name: str = "NSHRUNNER_SNAPSHOT_PATH"
    """The name of the environment variable to set the snapshot path to."""


def _tqdm_if_installed(iterable: Iterable[T], *args, **kwargs) -> Iterable[T]:
    try:
        from tqdm.auto import tqdm

        return cast(Iterable[T], tqdm(iterable, *args, **kwargs))
    except ImportError:
        return iterable


def _wrap_run_fn(
    config: Config,
    run_fn: Callable[[Unpack[TArguments]], TReturn],
    info_fn: Callable[[Unpack[TArguments]], RunInfo],
    validate_fn: Callable[[Unpack[TArguments]], tuple[Unpack[TArguments]] | None],
):
    @functools.wraps(run_fn)
    def wrapped_run_fn(*args: Unpack[TArguments]) -> TReturn:
        with contextlib.ExitStack() as stack:
            # Validate the configuration
            if (validate_out := validate_fn(*args)) is not None and isinstance(
                validate_out, tuple
            ):
                args = validate_out

            # Get the run info
            run_info = info_fn(*args)

            # Set up Python logging
            if not run_info.get("skip_python_logging", False):
                init_python_logging(config.python_logging)

            # Set additional environment variables
            env = {**(config.env or {}), **run_info.get("env", {})}
            stack.enter_context(_with_env(env))

            return run_fn(*args)

    return wrapped_run_fn


def _ensure_supports_session():
    # Make sure we have session installed
    try:
        subprocess.run(["session", "--version"], check=True)
    except FileNotFoundError:
        raise FileNotFoundError(
            "session is not installed. Please install session to use snapshot."
        )


def _launch_session(
    session_command: list[str],
    config_base_path: Path,
    session_name: str,
    attach: bool = True,
):
    return [
        "screen",
        "-dmS" if not attach else "-S",
        session_name,
        # Save the logs to a file
        "-L",
        "-Logfile",
        str((config_base_path / f"{session_name}.log").absolute()),
        # Enable UTF-8 encoding
        "-U",
        *session_command,
    ]


def _default_info_fn(*args: Unpack[TArguments]) -> RunInfo:
    return {}


def _default_validate_fn(*args: Unpack[TArguments]) -> None:
    pass


@dataclass(frozen=True)
class Runner(Generic[Unpack[TArguments], TReturn]):
    def generate_id(self):
        return str(uuid.uuid4())

    config: Config
    run_fn: Callable[[Unpack[TArguments]], TReturn]
    info_fn: Callable[[Unpack[TArguments]], RunInfo] = field(
        default_factory=lambda: _default_info_fn
    )
    validate_fn: Callable[[Unpack[TArguments]], tuple[Unpack[TArguments]] | None] = (
        field(default_factory=lambda: _default_validate_fn)
    )
    transform_fns: list[Callable[[Unpack[TArguments]], tuple[Unpack[TArguments]]]] = (
        field(default_factory=lambda: [])
    )

    def _transform(self, *args: Unpack[TArguments]) -> tuple[Unpack[TArguments]]:
        for transform_fn in self.transform_fns:
            args = transform_fn(*copy.deepcopy(args))
        return args

    @cached_property
    def _wrapped_run_fn(self):
        return _wrap_run_fn(
            self.config,
            self.run_fn,
            self.info_fn,
            self.validate_fn,
        )

    def with_transform(
        self,
        transform_fn: Callable[[Unpack[TArguments]], tuple[Unpack[TArguments]]],
    ):
        return replace(self, transform_fns=[*self.transform_fns, transform_fn])

    def _root_dir(self, runs: Sequence[tuple[Unpack[TArguments]]]):
        # If the user has provided a `savedir`, use that as the base path.
        if (savedir := self.config.savedir) is not None:
            return _gitignored_dir(Path(savedir) / "nshrunner", create=True)

        # If all configs have the same `project_root` config, use that instead.
        project_root_paths = set(self.info_fn(*args).get("base_dir") for args in runs)
        if (
            project_root_paths
            and len(project_root_paths) == 1
            and (project_root_path := project_root_paths.pop()) is not None
        ):
            project_root_path = Path(project_root_path)
        else:
            project_root_path = Path.cwd()

        return _gitignored_dir(project_root_path / "nshrunner", create=True)

    def _session_dir(
        self,
        runs: Sequence[tuple[Unpack[TArguments]]],
        id: str,
    ):
        root_dir = self._root_dir(runs)
        return _gitignored_dir(root_dir / id, create=True)

    def _resolve_runs(self, runs: Sequence[tuple[Unpack[TArguments]]]):
        # First, run all the transforms
        runs = [self._transform(*args) for args in runs]

        # Validate that there are no duplicate IDs
        if self.config.validate_no_duplicate_ids:
            ids = [
                id_
                for args in runs
                if (id_ := self.info_fn(*args).get("id")) is not None
            ]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate IDs found in the runs.")

        return runs

    def _resolve_env(self, env: Mapping[str, str] | None):
        return {**(self.config.env or {}), **(env or {})}

    def _setup_session(
        self,
        runs: Sequence[tuple[Unpack[TArguments]]],
        id: str | None = None,
        *,
        env: Mapping[str, str] | None,
        snapshot: SnapshotArgType,
    ):
        # Resolve all runs
        runs = self._resolve_runs(runs)

        # Create id if not provided
        if id is None:
            id = self.generate_id()

        # Create the session directory
        session_dir = self._session_dir(runs, id)

        # Resolve the environment
        env = self._resolve_env(env)

        # Take a snapshot of the environment if needed
        if (
            snapshot_config := SnapshotConfig._from_nshrunner_ctor(
                snapshot, configs=runs, base_dir=session_dir
            )
        ) is not None:
            snapshot_path = snapshot_modules(snapshot_config)
            # Update the environment to include the snapshot path
            env = {
                **env,
                self.config.snapshot_env_name: str(snapshot_path),
                "PYTHONPATH": f"{snapshot_path}:{os.environ.get('PYTHONPATH', '')}",
            }

        return runs, session_dir

    def local(self, runs: Sequence[tuple[Unpack[TArguments]]]):
        """
        Runs a list of configs locally.

        Parameters
        ----------
        runs : Sequence[tuple[Unpack[TArguments]]]
            A sequence of runs to run.
        """
        runs, _ = self._setup_session(runs)
        for args in _tqdm_if_installed(runs):
            yield self.run_fn(*args)

    def session(
        self,
        runs: Sequence[tuple[Unpack[TArguments]]],
        *,
        snapshot: SnapshotArgType,
        env: Mapping[str, str] | None = None,
    ):
        # Make sure the `session` utility is installed
        _ensure_supports_session()

        # Resolve all runs
        runs = self._resolve_runs(runs)

        base_dir = self._root_dir(runs)

        # Snapshot the environment
        if (
            snapshot_config := SnapshotConfig._from_nshrunner_ctor(
                snapshot, configs=runs, base_dir=base_dir
            )
        ) is not None:
            snapshot_modules(snapshot_config)

    def session(
        self,
        runs: Sequence[tuple[Unpack[TArguments]]],
        *,
        snapshot: bool | SnapshotConfig,
        name: str = "ll",
        env: Mapping[str, str] | None = None,
        setup_commands: Sequence[str] | None = None,
        activate_venv: bool = True,
        print_environment_info: bool = False,
        pause_before_exit: bool = False,
        attach: bool = True,
        print_command: bool = True,
        python_command_prefix: str | None = None,
    ):
        """
        Launches len(sessions) local runs in different environments using `screen`.

        Parameters
        ----------
        runs : Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]]
            A sequence of runs to launch.
        name : str, optional
            The name of this job. This name is pre-pended to the `screen` session names.
        env : Mapping[str, str], optional
            Environment variables to set for the session.
        snapshot : bool | SnapshotConfig
            Whether to snapshot the environment before launching the sessions.
        setup_commands : Sequence[str], optional
            A list of commands to run at the beginning of the shell script.
        activate_venv : bool, optional
            Whether to activate the virtual environment before running the jobs.
        print_environment_info : bool, optional
            Whether to print the environment information before starting each job.
        pause_before_exit : bool, optional
            Whether to pause before exiting the screen session.
        attach : bool, optional
            Whether to attach to the screen session after launching it.
        print_command : bool, optional
            Whether to print the command to the console.
        python_command_prefix : str, optional
            A prefix to add to the Python command. This would be used, for example, to run the Python command with a profiler (e.g., nsight-sys).
        """

        _ensure_supports_session()
        runs = self._resolve_runs(runs)

        # Generate a random ID for the session.
        # We'll use this ID for snapshotting, as well as for
        #   defining the name of the shell script that will launch the sessions.
        id = self.generate_id()

        # Resolve all runs
        resolved_runs = _resolve_runs(runs, validate=True)
        local_data_path = self._local_data_path(id, resolved_runs)

        # Setup commands and env
        setup_commands_pre: list[str] = []
        env = {**self.env, **(env or {})}

        # Handle snapshot
        snapshot_path = self._snapshot(snapshot, resolved_runs, local_data_path)
        if snapshot_path:
            snapshot_str = str(snapshot_path.resolve().absolute())
            setup_commands_pre.append(f"export {self.SNAPSHOT_ENV_NAME}={snapshot_str}")
            setup_commands_pre.append(f"export PYTHONPATH={snapshot_str}:$PYTHONPATH")

        # Conda environment
        if activate_venv:
            # Activate the conda environment
            setup_commands_pre.append("echo 'Activating environment'")
            setup_commands_pre.append(_shell_hook())

        setup_commands = setup_commands_pre + list(setup_commands or [])

        # Save all configs to pickle files
        from .picklerunner.create import serialize_many

        config_pickle_save_path = local_data_path / "sessions"
        config_pickle_save_path.mkdir(exist_ok=True)
        serialized = serialize_many(
            config_pickle_save_path,
            _runner_main,
            [
                ((self._run, self._init_kwargs, c, args), {})
                for c, args in resolved_runs
            ],
        )

        # Create the launcher script
        launcher_path = write_helper_script(
            config_pickle_save_path,
            serialized.bash_command_sequential(
                pause_before_exit=pause_before_exit,
                print_environment_info=print_environment_info,
            ),
            env,
            setup_commands,
            command_prefix=python_command_prefix,
            file_name="launcher.sh",
        )
        launcher_command = ["bash", str(launcher_path)]

        # Get the screen session command
        command = _launch_session(
            launcher_command,
            config_pickle_save_path,
            name,
            attach=attach,
        )
        command = " ".join(command)

        # Print the full command so the user can copy-paste it
        if print_command:
            print(f"Run the following command to launch the session:\n\n{command}")

        return command

    def _local_data_path(
        self,
        id: str,
        runs: list[tuple[TConfig, tuple[Unpack[TArguments]]]] | None = None,
    ) -> Path:
        # First, resolve the base path.
        base_path = self._get_base_path(runs)
        base_path.mkdir(parents=True, exist_ok=True)

        # Add a gitignore file to the directory so that the entire directory is ignored by git
        gitignore_path = base_path / ".gitignore"
        if not gitignore_path.exists():
            gitignore_path.touch()
            gitignore_path.write_text("*\n")

        local_data_path = base_path / id
        local_data_path.mkdir(exist_ok=True)

        return local_data_path

    @remove_lsf_environment_variables()
    @remove_slurm_environment_variables()
    @remove_wandb_environment_variables()
    def submit(
        self,
        runs: Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]],
        *,
        scheduler: unified.Scheduler | Literal["auto"] = "auto",
        snapshot: bool | SnapshotConfig,
        reset_id: bool = False,
        activate_venv: bool = True,
        print_environment_info: bool = False,
        env: Mapping[str, str] | None = None,
        print_command: bool = True,
        python_command_prefix: str | None = None,
        run_git_pre_commit_hook: bool = True,
        **kwargs: Unpack[unified.GenericJobKwargs],
    ):
        """
        Submits a list of runs to a cluster (SLURM or LSF).

        Parameters
        ----------
        runs : Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]]
            A sequence of runs to submit.
        scheduler : str, optional
            The scheduler to use. If `auto`, the scheduler will be inferred.
        snapshot : bool | Path
            The base path to save snapshots to. If `True`, a default path will be used.
        reset_id : bool, optional
            Whether to reset the id of the runs before launching them.
        activate_venv : bool, optional
            Whether to activate the virtual environment before running the jobs.
        print_environment_info : bool, optional
            Whether to print the environment information before starting each job.
        env : Mapping[str, str], optional
            Additional environment variables to set.
        print_command : bool, optional
            Whether to print the command to the console.
        python_command_prefix : str, optional
            A prefix to add to the Python command. This would be used, for example, to run the Python command with a profiler (e.g., nsight-sys).
        run_git_pre_commit_hook : bool, optional
            Whether to run the Git pre-commit hook before launching the sessions.
        kwargs : dict
            Additional keyword arguments to pass to the job submission script.
        """
        if run_git_pre_commit_hook:
            if not self._run_git_pre_commit_hook():
                raise ValueError("Git pre-commit hook failed. Aborting job submission.")

        if scheduler == "auto":
            scheduler = unified.infer_current_scheduler()
            log.critical(f"Inferred current scheduler as {scheduler}")

        id = self.generate_id()

        resolved_runs = _resolve_runs(runs, reset_id=reset_id, validate=True)
        local_data_path = self._local_data_path(id, resolved_runs)

        # Environment variables
        kwargs["environment"] = {
            **self.env,
            **kwargs.get("environment", {}),
            **(env or {}),
        }

        # Validate the submit options before proceeding
        unified.validate_kwargs(scheduler, kwargs)

        # Setup commands
        setup_commands_pre: list[str] = []

        # Handle snapshot
        snapshot_path = self._snapshot(snapshot, resolved_runs, local_data_path)
        if snapshot_path:
            snapshot_str = str(snapshot_path.resolve().absolute())
            setup_commands_pre.append(f"export {self.SNAPSHOT_ENV_NAME}={snapshot_str}")
            setup_commands_pre.append(f"export PYTHONPATH={snapshot_str}:$PYTHONPATH")

        # Conda environment
        if activate_venv:
            # Activate the conda environment
            setup_commands_pre.append("echo 'Activating environment'")
            setup_commands_pre.append(_shell_hook())

        kwargs["setup_commands"] = setup_commands_pre + list(
            kwargs.get("setup_commands", [])
        )

        base_path = local_data_path / "submit"
        base_path.mkdir(exist_ok=True, parents=True)

        # Serialize the runs
        map_array_args: list[
            tuple[
                RunProtocol[TConfig, TReturn, Unpack[TArguments]],
                Mapping[str, Any],
                TConfig,
                tuple[Unpack[TArguments]],
            ]
        ] = [(self._run, self._init_kwargs, c, args) for c, args in resolved_runs]
        submission = unified.to_array_batch_script(
            scheduler,
            base_path,
            _runner_main,
            map_array_args,
            print_environment_info=print_environment_info,
            python_command_prefix=python_command_prefix,
            **kwargs,
        )
        if print_command:
            print(
                f"Please run the following command to submit the jobs:\n\n{submission.command}"
            )

        return submission

    def submit_slurm(
        self,
        runs: Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]],
        *,
        snapshot: bool | SnapshotConfig,
        reset_id: bool = False,
        activate_venv: bool = True,
        print_environment_info: bool = False,
        env: Mapping[str, str] | None = None,
        print_command: bool = True,
        python_command_prefix: str | None = None,
        run_git_pre_commit_hook: bool = True,
        **kwargs: Unpack[unified.GenericJobKwargs],
    ):
        """
        Submits a list of runs to a SLURM cluster.

        Parameters
        ----------
        runs : Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]]
            A sequence of runs to submit.
        scheduler : str, optional
            The scheduler to use. If `auto`, the scheduler will be inferred.
        snapshot : bool | Path
            The base path to save snapshots to. If `True`, a default path will be used.
        reset_id : bool, optional
            Whether to reset the id of the runs before launching them.
        activate_venv : bool, optional
            Whether to activate the virtual environment before running the jobs.
        print_environment_info : bool, optional
            Whether to print the environment information before starting each job.
        env : Mapping[str, str], optional
            Additional environment variables to set.
        print_command : bool, optional
            Whether to print the command to the console.
        python_command_prefix : str, optional
            A prefix to add to the Python command. This would be used, for example, to run the Python command with a profiler (e.g., nsight-sys).
        run_git_pre_commit_hook : bool, optional
            Whether to run the Git pre-commit hook before launching the sessions.
        kwargs : dict
            Additional keyword arguments to pass to the job submission script.
        """
        return self.submit(
            runs,
            scheduler="slurm",
            snapshot=snapshot,
            reset_id=reset_id,
            activate_venv=activate_venv,
            print_environment_info=print_environment_info,
            env=env,
            print_command=print_command,
            python_command_prefix=python_command_prefix,
            run_git_pre_commit_hook=run_git_pre_commit_hook,
            **kwargs,
        )

    def submit_lsf(
        self,
        runs: Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]],
        *,
        snapshot: bool | SnapshotConfig,
        reset_id: bool = False,
        activate_venv: bool = True,
        print_environment_info: bool = False,
        env: Mapping[str, str] | None = None,
        print_command: bool = True,
        python_command_prefix: str | None = None,
        run_git_pre_commit_hook: bool = True,
        **kwargs: Unpack[unified.GenericJobKwargs],
    ):
        """
        Submits a list of runs to an LSF cluster.

        Parameters
        ----------
        runs : Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]]
            A sequence of runs to submit.
        scheduler : str, optional
            The scheduler to use. If `auto`, the scheduler will be inferred.
        snapshot : bool | Path
            The base path to save snapshots to. If `True`, a default path will be used.
        reset_id : bool, optional
            Whether to reset the id of the runs before launching them.
        activate_venv : bool, optional
            Whether to activate the virtual environment before running the jobs.
        print_environment_info : bool, optional
            Whether to print the environment information before starting each job.
        env : Mapping[str, str], optional
            Additional environment variables to set.
        print_command : bool, optional
            Whether to print the command to the console.
        python_command_prefix : str, optional
            A prefix to add to the Python command. This would be used, for example, to run the Python command with a profiler (e.g., nsight-sys).
        run_git_pre_commit_hook : bool, optional
            Whether to run the Git pre-commit hook before launching the sessions.
        kwargs : dict
            Additional keyword arguments to pass to the job submission script.
        """
        return self.submit(
            runs,
            scheduler="lsf",
            snapshot=snapshot,
            reset_id=reset_id,
            activate_venv=activate_venv,
            print_environment_info=print_environment_info,
            env=env,
            print_command=print_command,
            python_command_prefix=python_command_prefix,
            run_git_pre_commit_hook=run_git_pre_commit_hook,
            **kwargs,
        )


# First, let's create the function that's going to be run on the cluster.
def _runner_main(
    run_fn: RunProtocol[TConfig, TReturn, Unpack[TArguments]],
    runner_kwargs: Mapping[str, Any],
    config: TConfig,
    args: tuple[Unpack[TArguments]],
):
    # Create the runner
    runner = Runner(run_fn, **runner_kwargs)

    # Run the function and return the result
    return_values = runner.local([(config, *args)])
    assert len(return_values) == 1
    return return_values[0]


def _shell_hook():
    return f'eval "$(conda shell.bash hook)" && conda activate {sys.prefix}'
