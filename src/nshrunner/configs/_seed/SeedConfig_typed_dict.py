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
def CreateSeedConfig(dict: SeedConfigTypedDict, /) -> SeedConfig: ...


@typ.overload
def CreateSeedConfig(**dict: typ.Unpack[SeedConfigTypedDict]) -> SeedConfig: ...


def CreateSeedConfig(*args, **kwargs):
    from nshrunner._seed import SeedConfig

    dict = args[0] if args else kwargs
    return SeedConfig.model_validate(dict)
