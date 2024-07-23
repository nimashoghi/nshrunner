import datetime
import logging
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import nshconfig as C

log = logging.getLogger(__name__)

SNAPSHOT_DIR_NAME_DEFAULT = "nshrunner_snapshot"


class SnapshotConfig(C.Config):
    modules: list[str]
    """Additional modules to snapshot."""

    dir: Path
    """The directory to save snapshots to."""

    @classmethod
    def from_parent_modules(
        cls,
        configs: Sequence[Any],
        dir: Path | None = None,
        additional_modules: list[str] = [],
    ):
        modules_set: set[str] = set(additional_modules)

        for config in configs:
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
            modules_set.add(module)

        # Resolve dir
        if dir is None:
            id_ = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}--{str(uuid.uuid4())}"
            dir = Path.home() / ".cache" / "nshrunner" / "snapshots" / id_

        dir.mkdir(exist_ok=True, parents=True)
        return cls(modules=list(modules_set), dir=dir)
