import contextlib
import copy
import functools
import logging
import os
import sys
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Generic, TypeAlias, cast

import nshconfig as C
from typing_extensions import TypedDict, TypeVar, TypeVarTuple, Unpack

from . import _env
from ._logging import PythonLoggingConfig, init_python_logging
from ._seed import SeedConfig
from ._submit import lsf, screen, slurm
from ._util.env import _with_env
from ._util.environment import (
    remove_lsf_environment_variables,
    remove_nshrunner_environment_variables,
    remove_slurm_environment_variables,
    remove_wandb_environment_variables,
)
from ._util.git import _gitignored_dir
from ._util.signal_handling import (
    Signal,
    SignalHandler,
    SignalHandlers,
    _with_signal_handlers,
)
from .snapshot import SnapshotArgType, SnapshotConfig, SnapshotInfo, snapshot_modules

log = logging.getLogger(__name__)


_Path: TypeAlias = str | Path | os.PathLike


class RunInfo(TypedDict, total=False):
    id: str
    """The ID of the run."""

    base_dir: _Path | None
    """The base directory to save the run's files to."""

    env: Mapping[str, str]
    """Environment variables to set for the run."""

    skip_python_logging: bool
    """Whether to skip setting up Python logging for the run. Default: `False`."""


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

    snapshot: SnapshotInfo | None = None
    """The snapshot information for the session."""


class RunnerConfig(C.Config):
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


T = TypeVar("T", infer_variance=True)


def _tqdm_if_installed(iterable: Iterable[T], *args, **kwargs) -> Iterable[T]:
    try:
        from tqdm.auto import tqdm  # type: ignore

        return cast(Iterable[T], tqdm(iterable, *args, **kwargs))
    except ImportError:
        return iterable


def _wrap_run_fn(
    config: RunnerConfig,
    run_fn: Callable[[Unpack[TArguments]], TReturn],
    info_fn: Callable[[Unpack[TArguments]], RunInfo],
    validate_fns: Iterable[Callable[[Unpack[TArguments]], None]],
    signal_handlers: Mapping[Signal, Sequence[SignalHandler]],
):
    @functools.wraps(run_fn)
    def wrapped_run_fn(*args: Unpack[TArguments]) -> TReturn:
        # Validate the configuration
        for validate_fn in validate_fns:
            validate_fn(*args)

        # Get the run info
        run_info = info_fn(*args)

        # Set up Python logging
        if not run_info.get("skip_python_logging", False):
            init_python_logging(config.python_logging)

        # Set additional environment variables
        with contextlib.ExitStack() as stack:
            stack.enter_context(_with_env(run_info.get("env", {})))

            # Set up signal handlers
            if signal_handlers:
                stack.enter_context(_with_signal_handlers(signal_handlers))
                # TODO: implement this function

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
    Config: ClassVar = RunnerConfig

    def generate_id(self):
        return str(uuid.uuid4())

    @classmethod
    def default_info_fn(cls, *args: Unpack[TArguments]) -> RunInfo:
        return {}

    @classmethod
    def default_validate_fn(cls, *args: Unpack[TArguments]) -> None:
        pass

    def __init__(
        self,
        run_fn: Callable[[Unpack[TArguments]], TReturn],
        config: RunnerConfig = RunnerConfig(),
        *,
        info_fn: Callable[[Unpack[TArguments]], RunInfo] | None = None,
        validate_fn: Callable[[Unpack[TArguments]], None] | None = None,
        transform_fns: list[Callable[[Unpack[TArguments]], tuple[Unpack[TArguments]]]]
        | None = None,
        signal_handlers: SignalHandlers | None = None,
    ):
        self.config = config
        self.run_fn = run_fn
        self.info_fn = info_fn if info_fn is not None else self.default_info_fn
        self.validate_fn = (
            validate_fn if validate_fn is not None else self.default_validate_fn
        )
        self.transform_fns = transform_fns or []

        self.signal_handlers = defaultdict[Signal, list[SignalHandler]](lambda: [])
        if signal_handlers is not None:
            for signal_name, handlers in signal_handlers.items():
                for handler in handlers:
                    self.signal_handlers[signal_name].append(handler)

    def with_signal_handler(self, name: Signal, handler: SignalHandler):
        runner = copy.deepcopy(self)
        runner.signal_handlers[name].append(handler)
        return runner

    def transform_fn_generator(
        self,
        additional_transforms: list[
            Callable[[Unpack[TArguments]], tuple[Unpack[TArguments]]]
        ],
    ):
        yield from self.transform_fns
        yield from additional_transforms

    def _transform(
        self,
        *args: Unpack[TArguments],
        additional_transforms: list[
            Callable[[Unpack[TArguments]], tuple[Unpack[TArguments]]]
        ],
    ) -> tuple[Unpack[TArguments]]:
        for transform_fn in self.transform_fn_generator(additional_transforms):
            args = transform_fn(*copy.deepcopy(args))
        return args

    @property
    def _wrapped_run_fn(self) -> Callable[[Unpack[TArguments]], TReturn]:
        return _wrap_run_fn(
            self.config,
            self.run_fn,
            self.info_fn,
            (self.validate_fn,),
            self.signal_handlers,
        )

    def with_transform(
        self,
        transform_fn: Callable[[Unpack[TArguments]], tuple[Unpack[TArguments]]],
    ):
        runner = copy.deepcopy(self)
        runner.transform_fns.append(transform_fn)
        return runner

    def _root_dir(self, runs: Iterable[tuple[Unpack[TArguments]]]):
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
        runs: Iterable[tuple[Unpack[TArguments]]],
        id: str,
    ):
        root_dir = self._root_dir(runs)
        return _gitignored_dir(root_dir / id, create=True)

    def _resolve_runs(
        self,
        runs: Iterable[tuple[Unpack[TArguments]]],
        additional_transforms: list[
            Callable[[Unpack[TArguments]], tuple[Unpack[TArguments]]]
        ],
    ):
        # First, run all the transforms
        runs = [
            self._transform(*args, additional_transforms=additional_transforms)
            for args in runs
        ]

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

    def _setup_session(
        self,
        runs: Iterable[tuple[Unpack[TArguments]]],
        id: str | None = None,
        *,
        env: Mapping[str, str] | None,
        snapshot: SnapshotArgType,
        transforms: list[Callable[[Unpack[TArguments]], tuple[Unpack[TArguments]]]],
    ):
        # Resolve all runs
        runs = self._resolve_runs(runs, additional_transforms=transforms)

        # Create id if not provided
        if id is None:
            id = self.generate_id()

        # Create the session directory
        session_dir = self._session_dir(runs, id)

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
        if (
            snapshot_config := SnapshotConfig._from_nshrunner_ctor(
                snapshot, configs=runs, base_dir=session_dir
            )
        ) is not None:
            session.snapshot = snapshot_modules(snapshot_config)
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
        transforms: list[Callable[[Unpack[TArguments]], tuple[Unpack[TArguments]]]]
        | None = None,
    ):
        runs, session = self._setup_session(
            runs,
            env=env,
            snapshot=False,
            transforms=transforms or [],
        )

        with _with_env(session.env):
            for args in _tqdm_if_installed(runs):
                yield self._wrapped_run_fn(*args)

    def local(
        self,
        runs: Iterable[tuple[Unpack[TArguments]]],
        *,
        env: Mapping[str, str] | None = None,
        transforms: list[Callable[[Unpack[TArguments]], tuple[Unpack[TArguments]]]]
        | None = None,
    ):
        return list(self.local_generator(runs, env=env, transforms=transforms))

    @remove_nshrunner_environment_variables()
    @remove_lsf_environment_variables()
    @remove_slurm_environment_variables()
    @remove_wandb_environment_variables()
    def session(
        self,
        runs: Iterable[tuple[Unpack[TArguments]]],
        options: screen.ScreenJobKwargs = {},
        *,
        snapshot: SnapshotArgType,
        setup_commands: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        transforms: list[Callable[[Unpack[TArguments]], tuple[Unpack[TArguments]]]]
        | None = None,
        activate_venv: bool = True,
        print_command: bool = True,
    ):
        # Make sure the `session` utility is installed
        screen.ensure_has_screen()

        # Resolve all runs
        runs, session = self._setup_session(
            runs,
            env=env,
            snapshot=snapshot,
            transforms=transforms or [],
        )
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
    @remove_lsf_environment_variables()
    @remove_slurm_environment_variables()
    @remove_wandb_environment_variables()
    def submit_slurm(
        self,
        runs: Iterable[tuple[Unpack[TArguments]]],
        options: slurm.SlurmJobKwargs,
        *,
        snapshot: SnapshotArgType,
        setup_commands: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        transforms: list[Callable[[Unpack[TArguments]], tuple[Unpack[TArguments]]]]
        | None = None,
        activate_venv: bool = True,
        print_command: bool = True,
    ):
        # Resolve all runs
        runs, session = self._setup_session(
            runs,
            env=env,
            snapshot=snapshot,
            transforms=transforms or [],
        )
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

    @remove_nshrunner_environment_variables()
    @remove_lsf_environment_variables()
    @remove_slurm_environment_variables()
    @remove_wandb_environment_variables()
    def submit_lsf(
        self,
        runs: Iterable[tuple[Unpack[TArguments]]],
        options: lsf.LSFJobKwargs,
        *,
        snapshot: SnapshotArgType,
        setup_commands: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        transforms: list[Callable[[Unpack[TArguments]], tuple[Unpack[TArguments]]]]
        | None = None,
        activate_venv: bool = True,
        print_command: bool = True,
    ):
        # Resolve all runs
        runs, session = self._setup_session(
            runs,
            env=env,
            snapshot=snapshot,
            transforms=transforms or [],
        )
        base_dir = session.dir_path / "submit"
        base_dir.mkdir(parents=True, exist_ok=True)

        # Update the LSF options
        options = lsf.update_options(options, base_dir)

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
            environment={**session.env, **options.get("environment", {})},
            setup_commands=setup_commands,
            execution={"mode": "array"},
        )

        # Create the submission script
        submission = lsf.to_array_batch_script(
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
