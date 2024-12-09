from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cloudpickle
from typing_extensions import TypeAliasType


@dataclass
class SerializedArgs:
    """A serialized representation of function arguments."""

    path: Path
    """Path to the `picklerunner`-serialized arguments."""

    def load(self) -> tuple[Any, ...]:
        """Load the serialized arguments."""
        with self.path.open("rb") as f:
            return cloudpickle.load(f)


@dataclass
class SerializedCallable:
    """A serialized representation of a callable function."""

    path: Path
    """Path to the `picklerunner`-serialized callable function."""


MaybeSerializedCallable = TypeAliasType(
    "MaybeSerializedCallable", Callable | SerializedCallable
)

MaybeSerializedArgs = TypeAliasType(
    "MaybeSerializedArgs", tuple[Any, ...] | SerializedArgs
)
