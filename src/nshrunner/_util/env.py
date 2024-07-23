import contextlib
import importlib
import importlib.util
import os
import sys
from collections.abc import Mapping
from pathlib import Path


@contextlib.contextmanager
def _with_env(env: Mapping[str, str]):
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


@contextlib.contextmanager
def _with_pythonpath_prepend(*paths: tuple[Path, list[str]]):
    # Paths contains a list of tuples, where the first element is the path to prepend to sys.path
    # and the second element is a list of modules that are contained in that path.
    paths_old = []
    for path, modules in paths:
        # If the path is already in sys.path, we don't need to do anything
        if path in sys.path:
            continue

        # If the path is not in sys.path, we need to add it to sys.path
        sys.path.insert(0, str(path))
        paths_old.append((path, modules))

    try:
        # Reload the modules that we've added to sys.path
        for path, modules in paths_old:
            for module in modules:
                importlib.invalidate_caches()
                importlib.import_module(module)

        yield
    finally:
        for path, _ in paths_old:
            sys.path.remove(str(path))
