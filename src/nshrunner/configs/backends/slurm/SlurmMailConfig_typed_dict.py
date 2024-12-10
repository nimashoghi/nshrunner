from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from nshrunner.backends.slurm import SlurmMailConfig


__codegen__ = True

"""Configuration for email notifications."""


# Schema entries
class SlurmMailConfigTypedDict(typ.TypedDict, total=False):
    """Configuration for email notifications."""

    user: typ.Required[str]
    """Email address to receive SLURM notifications."""

    types: list[
        typ.Literal["NONE"]
        | typ.Literal["BEGIN"]
        | typ.Literal["END"]
        | typ.Literal["FAIL"]
        | typ.Literal["REQUEUE"]
        | typ.Literal["ALL"]
        | typ.Literal["INVALID_DEPEND"]
        | typ.Literal["STAGE_OUT"]
        | typ.Literal["TIME_LIMIT"]
        | typ.Literal["TIME_LIMIT_90"]
        | typ.Literal["TIME_LIMIT_80"]
        | typ.Literal["TIME_LIMIT_50"]
        | typ.Literal["ARRAY_TASKS"]
    ]
    """Types of events that should trigger email notifications
    
    Common values:
    - BEGIN: Job start
    - END: Job completion
    - FAIL: Job failure
    - TIME_LIMIT: Job reached time limit
    - ALL: All events."""


@typ.overload
def CreateSlurmMailConfig(
    **dict: typ.Unpack[SlurmMailConfigTypedDict],
) -> SlurmMailConfig: ...


@typ.overload
def CreateSlurmMailConfig(
    data: SlurmMailConfigTypedDict | SlurmMailConfig, /
) -> SlurmMailConfig: ...


def CreateSlurmMailConfig(*args, **kwargs):
    from nshrunner.backends.slurm import SlurmMailConfig

    if not args and kwargs:
        # Called with keyword arguments
        return SlurmMailConfig.from_dict(kwargs)
    elif len(args) == 1:
        return SlurmMailConfig.from_dict_or_instance(args[0])
    else:
        raise TypeError(
            f"CreateSlurmMailConfig accepts either a SlurmMailConfigTypedDict, "
            f"keyword arguments, or a SlurmMailConfig instance"
        )
