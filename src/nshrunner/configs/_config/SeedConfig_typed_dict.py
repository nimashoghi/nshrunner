from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from nshrunner._seed import SeedConfig


__codegen__ = True


# Schema entries
class SeedConfigTypedDict(typ.TypedDict, total=False):
    seed: typ.Required[int]
    """Seed for the random number generator."""

    seed_workers: bool
    """Whether to seed the workers of the dataloader (Only applicable to PyTorch Lightning)."""

    use_lightning: bool
    """Whether to use Lightning's seed_everything function (if available)."""


@typ.overload
def CreateSeedConfig(**dict: typ.Unpack[SeedConfigTypedDict]) -> SeedConfig: ...


@typ.overload
def CreateSeedConfig(data: SeedConfigTypedDict | SeedConfig, /) -> SeedConfig: ...


def CreateSeedConfig(*args, **kwargs):
    from nshrunner._seed import SeedConfig

    if not args and kwargs:
        # Called with keyword arguments
        return SeedConfig.from_dict(kwargs)
    elif len(args) == 1:
        return SeedConfig.from_dict_or_instance(args[0])
    else:
        raise TypeError(
            f"CreateSeedConfig accepts either a SeedConfigTypedDict, "
            f"keyword arguments, or a SeedConfig instance"
        )
