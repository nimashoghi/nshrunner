from __future__ import annotations

import contextlib
import os
from collections.abc import Mapping


@contextlib.contextmanager
def with_env(env: Mapping[str, str]):
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
