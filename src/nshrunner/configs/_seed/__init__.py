from __future__ import annotations

__codegen__ = True

from nshrunner._seed import SeedConfig as SeedConfig

from .SeedConfig_typed_dict import CreateSeedConfig as CreateSeedConfig
from .SeedConfig_typed_dict import SeedConfigTypedDict as SeedConfigTypedDict

SeedConfigInstanceOrDict = SeedConfig | SeedConfigTypedDict


__all__ = [
    "CreateSeedConfig",
    "SeedConfig",
    "SeedConfigInstanceOrDict",
    "SeedConfigTypedDict",
]
