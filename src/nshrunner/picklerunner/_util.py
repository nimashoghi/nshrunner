from collections.abc import Callable, Mapping, Sequence
from typing import Any

from typing_extensions import TypedDict


class SerializedFunctionCallDict(TypedDict):
    fn: Callable
    args: Sequence[Any]
    kwargs: Mapping[str, Any]


JOB_INDEX_ENV_VAR = "NSHRUNNER_JOB_INDEX"
