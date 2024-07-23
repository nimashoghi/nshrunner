import datetime
import logging
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeAlias

import nshconfig as C
from typing_extensions import TypedDict, assert_never

from .._util.git import _gitignored_dir

log = logging.getLogger(__name__)

SNAPSHOT_DIR_NAME_DEFAULT = "nshrunner_snapshots"


class SnapshotConfigDict(TypedDict, total=False):
    dir: Path
    """The directory to save snapshots to. Default: `~/.cache/nshrunner/snapshots/{timestamp}--{uuid}`."""

    parent_modules: bool
    """Whether to snapshot the parent modules of the provided modules. Default: `True`."""

    modules: list[str]
    """Additional modules to snapshot. Default: `[]`."""


SnapshotArgType: TypeAlias = bool | SnapshotConfigDict


class SnapshotConfig(C.Config):
    modules: list[str] = []
    """Modules to snapshot."""

    dir: Path
    """The directory to save snapshots to."""

    @classmethod
    def _from_nshrunner_ctor(
        cls,
        value: SnapshotArgType,
        *,
        configs: Sequence[Any],
        base_dir: Path | None = None,
    ):
        match value:
            case False:
                return None
            case True:
                modules = [*_resolve_parent_modules(configs)]
            case Mapping():
                modules = value.get("modules", [])
                if value.get("parent_modules", True):
                    modules = [*modules, *_resolve_parent_modules(configs)]
            case _:
                assert_never(value)
        return cls(modules=modules, dir=_resolve_dir(base_dir))


def _resolve_parent_modules(configs: Sequence[Any]):
    modules_set = set[str]()

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

    return list(modules_set)


def _resolve_dir(base_dir: Path | None = None):
    # Resolve dir
    if base_dir is None:
        base_dir = Path.home() / ".cache" / "nshrunner"
        base_dir.mkdir(exist_ok=True, parents=True)

    id_ = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}--{str(uuid.uuid4())}"
    dir_ = _gitignored_dir(_gitignored_dir(base_dir / "snapshots") / id_)
    return dir_
