from __future__ import annotations

import os
import subprocess
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

import cloudpickle
import nshconfig as C
from typing_extensions import override

from ..picklerunner import SerializedArgs, SerializedCallable, execute
from .base import BaseBackend, BaseBackendConfig, Job, JobInfo, JobStatus, JobStatusInfo


class ScreenJob(Job):
    def __init__(self, job_id: str, work_dir: Path, num_tasks: int):
        self._job_id = job_id
        self._work_dir = work_dir
        self._num_tasks = num_tasks

        # Results will be stored in a file named 'results.pkl'
        # We consider a job done if that file exists.
        self._result_path = self._work_dir / "results.pkl"

    @property
    def info(self) -> JobInfo:
        return JobInfo(id=self._job_id)

    def status(self) -> JobStatusInfo:
        # Check if the results file is there
        if self._result_path.exists():
            return JobStatusInfo(status=JobStatus.COMPLETED)

        # Check if screen session is still running
        # If no session with self._job_id, then it may have failed
        if _screen_session_exists(self._job_id):
            return JobStatusInfo(status=JobStatus.RUNNING)
        else:
            # No results, no session = failed or exited with error
            return JobStatusInfo(status=JobStatus.FAILED)

    def results(self) -> list[Any]:
        # Block until results are ready
        while not self._result_path.exists():
            if self.status().status not in ["RUNNING"]:
                # If job is no longer running but no results file found, it's a failure
                raise RuntimeError(f"Job {self._job_id} failed or no results found.")
            time.sleep(0.5)

        with self._result_path.open("rb") as f:
            results = cloudpickle.load(f)
        return results


class ScreenBackendConfig(BaseBackendConfig):
    pass


class ScreenBackend(BaseBackend):
    """
    ScreenBackend executes jobs using GNU screen. It handles serialization of the function
    and arguments, executes them inside a screen session, and stores results in a file.
    """

    def __init__(
        self,
        base_dir: Path | str = ".",
        python_executable: str = "python",
    ):
        self._base_dir = Path(base_dir).resolve()
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._python = python_executable

    @override
    def execute_impl(
        self,
        fn: SerializedCallable,
        args_list: Sequence[SerializedArgs],
    ) -> Job:
        """
        Execute multiple jobs in parallel. For simplicity, we launch multiple screen sessions,
        one per job. Another approach would be to invoke them as an array within a single screen.
        """
        job_id = str(uuid.uuid4())
        job_dir = self._base_dir / job_id
        job_dir.mkdir()

        # We create one command per set of arguments and run them all in separate screens.
        # Results from each job are collected and stored as a list.
        # We'll have each job write its result to a temporary file, and then combine them.
        partial_results = []
        for i, arg in enumerate(args_list):
            partial_id = f"{job_id}_{i}"
            partial_dir = job_dir / f"task_{i}"
            partial_dir.mkdir()

            # We'll run each sub-task
            self._launch_single_task(fn, arg, partial_id, partial_dir, self._python)

            partial_results.append(partial_dir / "results.pkl")

        # We'll create a "wait script" that waits for all partial results to appear, then merges them
        wait_script = job_dir / "wait_and_merge_results.sh"
        with wait_script.open("w") as f:
            f.write("#!/bin/bash\n")
            f.write("set -e\n")
            # Wait for all partial results
            for i, pr in enumerate(partial_results):
                f.write(f'while [ ! -f "{pr}" ]; do sleep 0.5; done\n')
            # Once all partial results exist, merge them
            f.write(f"python -c 'import cloudpickle; import sys;\n")
            f.write("results = []\n")
            for pr in partial_results:
                f.write(f'with open("{pr}", "rb") as pf:\n')
                f.write(f"    results.extend(cloudpickle.load(pf))\n")
            final_path = job_dir / "results.pkl"
            f.write(f'with open("{final_path}", "wb") as outf:\n')
            f.write(f"    cloudpickle.dump(results, outf)\n'\n")
        wait_script.chmod(0o755)

        # Launch a screen session to wait and merge results
        # We only attach if needed. Usually for batch, we do not attach.
        subprocess.run(
            [
                "screen",
                "-dmS",
                job_id,
                "-U",
                "bash",
                str(wait_script),
            ],
            check=True,
        )

        return ScreenJob(job_id=job_id, work_dir=job_dir, num_tasks=len(args_list))

    def execute_single(
        self,
        fn: SerializedCallable,
        args: SerializedArgs,
    ) -> Job:
        """Execute a single job in a single screen session."""
        job_id = str(uuid.uuid4())
        job_dir = self._base_dir / job_id
        job_dir.mkdir()

        self._launch_single_task(fn, args, job_id, job_dir, self._python)

        return ScreenJob(job_id=job_id, work_dir=job_dir, num_tasks=1)

    def execute_sequential(
        self,
        fn: SerializedCallable,
        args: Sequence[SerializedArgs],
    ) -> Job:
        """
        Execute multiple arguments sequentially in one screen session.
        We'll run them one by one and aggregate the results into a single file.
        """
        job_id = str(uuid.uuid4())
        job_dir = self._base_dir / job_id
        job_dir.mkdir()

        seq_script = job_dir / "sequential_run.sh"
        with seq_script.open("w") as f:
            f.write("#!/bin/bash\n")
            f.write("set -e\n")
            f.write("RESULTS=()\n")
            for i, arg in enumerate(args):
                f.write(f'echo "Running task {i}"\n')
                f.write(
                    f"{self._python} -m nshrunner._src.picklerunner {fn.path} {arg.path} > {job_dir}/task_{i}.out\n"
                )
                f.write(f'RET=$(cat {job_dir}/task_{i}.out | sed -n "s/Result: //p")\n')
                # We'll deserialize the result if needed, but since picklerunner prints the result,
                # we assume it's picklable. If it's a complex object, consider using picklerunner differently.
                # For simplicity, assume it's a string. For a real scenario, adjust logic to handle arbitrary returns.
                f.write(
                    "python -c \"import sys,cloudpickle; cloudpickle.dump(['$RET'], open('{}/task_{}.pkl','wb'))\"\n".format(
                        job_dir, i
                    )
                )
                f.write("\n")

            # Now merge all partial results
            f.write(
                "python -c \"import cloudpickle; import sys; import os; results=[]; [results.extend(cloudpickle.load(open(os.path.join('{}', f))) ) for f in os.listdir('{}') if f.endswith('.pkl')]; cloudpickle.dump(results, open('{}/results.pkl', 'wb'))\"\n".format(
                    job_dir, job_dir, job_dir
                )
            )

        seq_script.chmod(0o755)

        subprocess.run(
            [
                "screen",
                "-dmS",
                job_id,
                "-U",
                "bash",
                str(seq_script),
            ],
            check=True,
        )

        return ScreenJob(job_id=job_id, work_dir=job_dir, num_tasks=len(args))

    def _launch_single_task(
        self,
        fn: SerializedCallable,
        args: SerializedArgs,
        job_id: str,
        work_dir: Path,
        python: str,
    ):
        """Helper to launch a single picklerunner call in screen."""
        run_script = work_dir / "run.sh"
        with run_script.open("w") as f:
            f.write("#!/bin/bash\n")
            f.write("set -e\n")
            # Run the picklerunner command
            f.write(
                f"{python} -m nshrunner._src.picklerunner {fn.path} {args.path} > {work_dir}/task.out\n"
            )
            f.write(
                'RET=$(cat {}/task.out | sed -n "s/Result: //p")\n'.format(work_dir)
            )
            f.write(
                "python -c \"import sys,cloudpickle; cloudpickle.dump(['$RET'], open('{}/results.pkl','wb'))\"\n".format(
                    work_dir
                )
            )
        run_script.chmod(0o755)

        # Launch screen session
        subprocess.run(
            [
                "screen",
                "-dmS",
                job_id,
                "-U",
                "bash",
                str(run_script),
            ],
            check=True,
        )


def _screen_session_exists(session_name: str) -> bool:
    # Check if a screen session with session_name exists
    # `screen -ls` lists sessions. If session_name not in output, session doesn't exist.
    try:
        output = subprocess.check_output(["screen", "-ls"]).decode("utf-8")
        return session_name in output
    except subprocess.CalledProcessError:
        return False
