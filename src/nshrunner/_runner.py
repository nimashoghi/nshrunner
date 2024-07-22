import contextlib
import copy
import logging
import os
import shutil
import subprocess
import sys
import traceback
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from functools import wraps
from pathlib import Path
from typing import Any, Generic, Literal, Protocol, cast, runtime_checkable

from typing_extensions import TypedDict, TypeVar, TypeVarTuple, Unpack, override

from ._submit.session import unified
from ._submit.session._script import create_launcher_script_file
from ._util import seed
from ._util.environment import (
    remove_lsf_environment_variables,
    remove_slurm_environment_variables,
    remove_wandb_environment_variables,
)
from .log import init_python_logging
from .model.config import BaseConfig
from .snapshot import snapshot_modules
from .trainer import Trainer

log = logging.getLogger(__name__)


def _tqdm_if_installed(iterable, *args, **kwargs):
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, *args, **kwargs)
    except ImportError:
        return iterable


class SnapshotConfig(TypedDict, total=False):
    dir: Path
    """The directory to save snapshots to. Default: `{cwd}/ll-{id}/snapshot`."""

    snapshot_ll: bool
    """Whether to snapshot the `ll` module. Default: `True`."""

    snapshot_config_cls_module: bool
    """Whether to snapshot the module of the config class. Default: `True`."""

    modules: list[str]
    """Additional modules to snapshot. Default: `[]`."""


SNAPSHOT_CONFIG_DEFAULT = SnapshotConfig(
    snapshot_ll=False,
    snapshot_config_cls_module=True,
)


TConfig = TypeVar("TConfig", bound=BaseConfig, infer_variance=True)
TReturn = TypeVar("TReturn", default=None, infer_variance=True)
TArguments = TypeVarTuple("TArguments", default=Unpack[tuple[()]])


def _validate_runs(runs: list[tuple[TConfig, tuple[Unpack[TArguments]]]]):
    if not runs:
        raise ValueError("No run configs provided.")

    # Make sure there are no duplicate ids
    id_counter = Counter(config.id for config, _ in runs if config.id is not None)
    duplicate_ids = {id for id, count in id_counter.items() if count > 1}
    if duplicate_ids:
        raise ValueError(
            f"Duplicate run IDs found: {duplicate_ids}. Each run must have a unique ID."
        )


def _resolve_run(
    run: TConfig | tuple[TConfig, Unpack[TArguments]],
    copy_config: bool = True,
    reset_id: bool = False,
) -> tuple[TConfig, tuple[Unpack[TArguments]]]:
    if isinstance(run, tuple):
        (config, *args) = run
    else:
        config = cast(TConfig, run)
        args = ()
    args = cast(tuple[Unpack[TArguments]], args)
    if copy_config:
        config = copy.deepcopy(config)
    if reset_id:
        config.id = BaseConfig.generate_id(ignore_rng=True)
    return (config, args)


def _resolve_runs(
    runs: Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]],
    copy_config: bool = True,
    reset_id: bool = False,
    validate: bool = False,
):
    resolved: list[tuple[TConfig, tuple[Unpack[TArguments]]]] = []
    for run in runs:
        resolved.append(_resolve_run(run, copy_config=copy_config, reset_id=reset_id))

    if validate:
        _validate_runs(resolved)

    return resolved


@runtime_checkable
class RunProtocol(Protocol[TConfig, TReturn, Unpack[TArguments]]):
    def __call__(self, config: TConfig, *args: Unpack[TArguments]) -> TReturn: ...


class Runner(Generic[TConfig, TReturn, Unpack[TArguments]]):
    DEFAULT_ENV: dict[str, str] = {}
    SNAPSHOT_ENV_NAME = "LL_SNAPSHOT"

    @classmethod
    def active_snapshot(cls) -> Path | None:
        if (snapshot := os.environ.get(cls.SNAPSHOT_ENV_NAME)) is not None:
            return Path(snapshot)
        return None

    @override
    def __init__(
        self,
        run: RunProtocol[TConfig, TReturn, Unpack[TArguments]],
        *,
        savedir: str | Path | os.PathLike | None = None,
        job_name: str = "ll",
        validate_config_before_run: bool = True,
        validate_strict: bool = True,
        env: Mapping[str, str] | None = None,
    ):
        """This is the initialization function for a class that takes in a run protocol, an auto wrap run
        boolean, and a slurm job name string.

        Parameters
        ----------
        run : RunProtocol[TConfig, Unpack[TArguments]]
            `run` is an instance of a class that implements the `RunProtocol` interface. It represents the main function or entry point of the program that will be executed.
        savedir : Path, optional
            The `savedir` parameter is a string that represents the directory where the program will save its execution files and logs.
            This is used when submitting the program to a SLURM/LSF cluster or when using the `local_sessions` method.
            If `None`, this will default to the current working directory / `llrunner`.
        job_name : str, optional
            The `job_name` parameter is a string that represents the name of the job when submitting it to a cluster.
        validate_config_before_run : bool, optional
            The `validate_config_before_run` parameter is a boolean that represents whether or not to validate the configuration before running the program.
        validate_strict: bool, optional
            Should `validate_config_before_run` be strict? If `True`, the configuration will be validated strictly. If `False`, the configuration will be validated non-strictly.
        """

        super().__init__()

        self._run = run
        self._savedir = savedir
        self.job_name = job_name
        self.validate_config_before_run = validate_config_before_run
        self.validate_strict = validate_strict
        self._init_kwargs = {
            "savedir": savedir,
            "job_name": job_name,
            "validate_config_before_run": validate_config_before_run,
            "validate_strict": validate_strict,
        }
        self.env: dict[str, str] = {
            **self.DEFAULT_ENV,
            **(env or {}),
        }

    def _run_git_pre_commit_hook(self):
        git_dir = self._find_git_dir()
        if not git_dir:
            log.info("Not a git repository. Skipping pre-commit hook.")
            return True

        pre_commit_hook = git_dir / "hooks" / "pre-commit"
        if not pre_commit_hook.exists():
            log.info("No pre-commit hook found. Skipping.")
            return True

        try:
            result = subprocess.run(
                [str(pre_commit_hook)],
                check=True,
                capture_output=True,
                text=True,
                cwd=git_dir.parent,
            )
            log.info("Git pre-commit hook passed successfully.")
            return True
        except subprocess.CalledProcessError as e:
            log.error(f"Git pre-commit hook failed. Output:\n{e.stdout}\n{e.stderr}")
            return False

    def _find_git_dir(self):
        current_dir = Path.cwd()
        while current_dir != current_dir.parent:
            git_dir = current_dir / ".git"
            if git_dir.is_dir():
                return git_dir
            current_dir = current_dir.parent
        return None

    def _get_base_path(
        self,
        runs: list[tuple[TConfig, tuple[Unpack[TArguments]]]] | None,
    ):
        # If the user has provided a `savedir`, use that as the base path.
        if self._savedir is not None:
            base_path = Path(self._savedir)
            base_path.mkdir(exist_ok=True, parents=True)
            return base_path

        # If all configs have the same `project_root` config, use that instead.
        project_root_paths = set(
            str(project_root.absolute())
            if (project_root := config.directory.project_root) is not None
            else None
            for config, _ in (runs or [])
        )
        if (
            project_root_paths
            and len(project_root_paths) == 1
            and (project_root_path := project_root_paths.pop()) is not None
        ):
            project_root_path = Path(project_root_path)
        else:
            project_root_path = Path.cwd()

        base_path = project_root_path / "llrunner"
        base_path.mkdir(exist_ok=True, parents=True)

        return base_path

    def _setup_python_logging(self, root_config: BaseConfig):
        """
        Sets up the logger with the specified configurations.

        Args:
            root_config (BaseConfig): The root configuration object.
        """
        config = root_config.runner.python_logging

        return init_python_logging(
            lovely_tensors=config.lovely_tensors,
            lovely_numpy=config.lovely_numpy,
            rich=config.rich,
            log_level=config.log_level,
            log_save_dir=root_config.directory.resolve_subdirectory(
                root_config.id, "stdio"
            ),
        )

    def _dump_run_information(self, config: BaseConfig):
        try:
            import yaml

        except ImportError:
            log.warning("Failed to import `yaml`. Skipping dumping of run information.")
            return

        dump_dir = config.directory.resolve_subdirectory(config.id, "stdio") / "dump"

        # Create a different directory for each rank.
        # Easy way for now: Add a random subdir.
        dump_dir = dump_dir / f"rank_{str(uuid.uuid4())}"
        dump_dir.mkdir(parents=True, exist_ok=True)

        # First, dump the full config
        full_config_path = dump_dir / "config.yaml"
        config_dict = config.model_dump(mode="json")
        with full_config_path.open("w") as file:
            yaml.dump(config_dict, file)

        # Dump all environment variables
        env_vars_path = dump_dir / "env.yaml"
        env_vars = dict(os.environ)
        with env_vars_path.open("w") as file:
            yaml.dump(env_vars, file)

        # Dump the output of `nvidia-smi` to a file (if available)
        # First, resolve either `nvidia-smi` or `rocm-smi` (for AMD GPUs)
        if not (smi_exe := self._resolve_gpu_smi()):
            return

        nvidia_smi_path = dump_dir / "nvidia_smi_output.log"
        try:
            with nvidia_smi_path.open("w") as file:
                subprocess.run([smi_exe], stdout=file, stderr=subprocess.PIPE)
        except FileNotFoundError:
            log.warning(f"Failed to run `{smi_exe}`.")

    def _resolve_gpu_smi(self):
        if shutil.which("nvidia-smi"):
            return "nvidia-smi"
        elif shutil.which("rocm-smi"):
            return "rocm-smi"
        else:
            log.warning("No GPU monitoring tool found.")
            return None

    @property
    def _run_fn(self) -> RunProtocol[TConfig, TReturn, Unpack[TArguments]]:
        run = self._run

        @wraps(run)
        def wrapped_run(config: TConfig, *args: Unpack[TArguments]) -> TReturn:
            nonlocal self

            with ExitStack() as stack:
                nonlocal run

                # If `validate_config_before_run`, we validate the configuration before running the program.
                if self.validate_config_before_run:
                    config = config.model_deep_validate(strict=self.validate_strict)

                # Set additional environment variables
                if additional_env := config.runner.additional_env_vars:
                    stack.enter_context(self._with_env(additional_env))

                # Set up Python logging
                self._setup_python_logging(config)

                # Seed everything
                seed.seed_everything(
                    config.runner.seed.seed,
                    workers=config.runner.seed.seed_workers,
                )

                # Auto-wrap the run in a Trainer context
                if config.trainer.auto_wrap_trainer:
                    stack.enter_context(Trainer.context(config))
                    log.info("Auto-wrapping run in Trainer context")

                # Dump run information
                if config.runner.dump_run_information:
                    self._dump_run_information(config)

                return run(config, *args)

            raise RuntimeError("ExitStack should never raise an exception")

        return wrapped_run

    @contextlib.contextmanager
    def _with_env(self, env: Mapping[str, str]):
        env_old = {k: os.environ.get(k, None) for k in env}
        os.environ.update(env)
        try:
            yield
        finally:
            for new_env_key in env.keys():
                # If we didn't have the key before, remove it
                if (old_value := env_old.get(new_env_key)) is None:
                    _ = os.environ.pop(new_env_key, None)
                else:
                    os.environ[new_env_key] = old_value

    def local(
        self,
        runs: Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]],
        env: Mapping[str, str] | None = None,
    ):
        """
        Runs a list of configs locally.

        Parameters
        ----------
        runs : Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]]
            A sequence of runs to submit.
        env : Mapping[str, str], optional
            Additional environment variables to set.
        """
        return_values: list[TReturn] = []
        for run in runs:
            config, args = _resolve_run(run)
            with self._with_env(env or {}):
                return_value = self._run_fn(config, *args)
                return_values.append(return_value)

        return return_values

    def fast_dev_run(
        self,
        runs: Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]],
        n_batches: int = 1,
        *,
        gpus: Sequence[int] | None = None,
        env: Mapping[str, str] | None = None,
        stop_on_error: bool = True,
        reset_memory_caches: bool = True,
        reset_ids: bool = True,
    ):
        """
        Runs a list of configs locally with `LightningTrainer.fast_dev_run = True`.

        Parameters
        ----------
        runs : Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]]
            A sequence of runs to submit.
        n_batches : int, optional
            The number of batches to run for `fast_dev_run`.
        gpus : Sequence[int], optional
            The GPUs to use for the runs.
        env : Mapping[str, str], optional
            Additional environment variables to set.
        stop_on_error : bool, optional
            Whether to stop on error.
        reset_memory_caches : bool, optional
            Whether to reset memory caches after each run.
        reset_ids : bool, optional
            Whether to reset the id of the runs before running them. This prevents the
            dev runs' logs from overwriting the main runs' logs.
        """
        resolved_runs = _resolve_runs(
            runs, copy_config=True, reset_id=reset_ids, validate=True
        )

        return_values: list[TReturn] = []
        with self._with_env(env or {}):
            for config, args in _tqdm_if_installed(resolved_runs, desc="Fast dev run"):
                run_id = config.id
                run_name = config.run_name
                try:
                    if gpus is not None:
                        config.trainer.accelerator = "gpu"
                        config.trainer.devices = gpus
                    config.trainer.fast_dev_run = n_batches
                    return_values.append(self._run_fn(config, *args))
                except BaseException as e:
                    log.critical(f"Error in run with {run_id=} ({run_name=}): {e}")
                    if stop_on_error:
                        raise
                    else:
                        # Print full traceback
                        traceback.print_exc()
                finally:
                    # After each run, we should reset memory/caches
                    if reset_memory_caches:
                        self._reset_memory_caches()

        return return_values

    def _launch_session(
        self,
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

    def session(
        self,
        runs: Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]],
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
        run_git_pre_commit_hook: bool = True,
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
        run_git_pre_commit_hook : bool, optional
            Whether to run the Git pre-commit hook before launching the sessions.
        """

        if run_git_pre_commit_hook:
            if not self._run_git_pre_commit_hook():
                raise ValueError("Git pre-commit hook failed. Aborting session launch.")

        # Generate a random ID for the session.
        # We'll use this ID for snapshotting, as well as for
        #   defining the name of the shell script that will launch the sessions.
        id = str(uuid.uuid4())

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
        from .picklerunner import serialize_many

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
        launcher_path = config_pickle_save_path / "launcher.sh"
        create_launcher_script_file(
            launcher_path,
            serialized.bash_command_sequential(
                pause_before_exit=pause_before_exit,
                print_environment_info=print_environment_info,
            ),
            env,
            setup_commands,
            command_prefix=python_command_prefix,
        )
        launcher_command = ["bash", str(launcher_path)]

        # Get the screen session command
        command = self._launch_session(
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

    def fast_dev_run_session(
        self,
        runs: Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]],
        n_batches: int = 1,
        *,
        snapshot: bool | SnapshotConfig = False,
        gpus: Sequence[int] | None = None,
        env: Mapping[str, str] | None = None,
        setup_commands: Sequence[str] | None = None,
        activate_venv: bool = True,
        print_environment_info: bool = False,
        pause_before_exit: bool = False,
        attach: bool = True,
        print_command: bool = True,
        reset_ids: bool = True,
        python_command_prefix: str | None = None,
        run_git_pre_commit_hook: bool = True,
    ):
        """
        Runs a list of configs locally with `LightningTrainer.fast_dev_run = True`.

        Parameters
        ----------
        runs : Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]]
            A sequence of runs to submit.
        n_batches : int, optional
            The number of batches to run for `fast_dev_run`.
        snapshot : bool | Path, optional
            The base path to save snapshots to. If `True`, a default path will be used. If `False`, no snapshots will be taken.
        gpus : Sequence[int], optional
            The GPUs to use for the runs.
        env : Mapping[str, str], optional
            Additional environment variables to set.
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
        reset_ids : bool, optional
            Whether to reset the id of the runs before running them. This prevents the
            dev runs' logs from overwriting the main runs' logs.
        python_command_prefix : str, optional
            A prefix to add to the Python command. This would be used, for example, to run the Python command with a profiler (e.g., nsight-sys).
        run_git_pre_commit_hook : bool, optional
            Whether to run the Git pre-commit hook before launching the sessions.
        """
        resolved_runs = _resolve_runs(
            runs, copy_config=True, reset_id=reset_ids, validate=True
        )
        for config, _ in resolved_runs:
            config.trainer.fast_dev_run = n_batches
            if gpus is not None:
                config.trainer.accelerator = "gpu"
                config.trainer.devices = gpus

        return self.session(
            [(configs, *args) for configs, args in resolved_runs],
            snapshot=snapshot,
            name="ll-fast_dev_run",
            env=env,
            setup_commands=setup_commands,
            attach=attach,
            print_command=print_command,
            activate_venv=activate_venv,
            print_environment_info=print_environment_info,
            pause_before_exit=pause_before_exit,
            python_command_prefix=python_command_prefix,
            run_git_pre_commit_hook=run_git_pre_commit_hook,
        )

    def _reset_memory_caches(self):
        import gc

        import torch

        # Clear the memory caches
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.synchronize()
        gc.collect()

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

    def _snapshot(
        self,
        snapshot: bool | SnapshotConfig,
        resolved_runs: list[tuple[TConfig, tuple[Unpack[TArguments]]]],
        local_data_path: Path,
    ):
        # Handle snapshot
        snapshot_config: SnapshotConfig | None = None
        if snapshot is True:
            snapshot_config = {**SNAPSHOT_CONFIG_DEFAULT}
        elif snapshot is False:
            snapshot_config = None
        elif isinstance(snapshot, Mapping):
            snapshot_config = {**SNAPSHOT_CONFIG_DEFAULT, **snapshot}

        del snapshot
        if snapshot_config is None:
            return None

        # Set the snapshot base to the user's home directory
        snapshot_dir = snapshot_config.get("dir", local_data_path / "snapshot")
        snapshot_dir.mkdir(exist_ok=True, parents=True)

        snapshot_modules_set: set[str] = set()
        snapshot_modules_set.update(snapshot_config.get("modules", []))
        if snapshot_config.get("snapshot_ll", True):
            # Resolve ll by taking the module of the runner class
            ll_module = self.__class__.__module__.split(".", 1)[0]
            if ll_module != "ll":
                log.warning(
                    f"Runner class {self.__class__.__name__} is not in the 'll' module.\n"
                    "This is unexpected and may lead to issues with snapshotting."
                )
            snapshot_modules_set.add(ll_module)
        if snapshot_config.get("snapshot_config_cls_module", True):
            for config, _ in resolved_runs:
                # Resolve the root module of the config class
                # NOTE: We also must handle the case where the config
                #   class's module is "__main__" (i.e. the config class
                #   is defined in the main script).
                module = config.__class__.__module__
                if module == "__main__":
                    log.warning(
                        f"Config class {config.__class__.__name__} is defined in the main script.\n"
                        "Snapshotting the main script is not supported.\n"
                        "Skipping snapshotting of the config class's module."
                    )
                    continue

                # Make sure to get the root module
                module = module.split(".", 1)[0]
                snapshot_modules_set.add(module)

        snapshot_path = snapshot_modules(snapshot_dir, list(snapshot_modules_set))
        return snapshot_path.absolute()

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

        id = str(uuid.uuid4())

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
