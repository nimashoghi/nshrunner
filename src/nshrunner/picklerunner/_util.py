from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from typing_extensions import TypedDict


class SerializedFunctionCallDict(TypedDict):
    fn: Callable
    args: Sequence[Any]
    kwargs: Mapping[str, Any]
