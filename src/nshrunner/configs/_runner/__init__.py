from __future__ import annotations

__codegen__ = True

from nshrunner._runner import Config as Config
from nshrunner._runner import SeedConfig as SeedConfig

from .Config_typed_dict import ConfigTypedDict as ConfigTypedDict
from .Config_typed_dict import CreateConfig as CreateConfig

ConfigInstanceOrDict = Config | ConfigTypedDict

from .SeedConfig_typed_dict import CreateSeedConfig as CreateSeedConfig
from .SeedConfig_typed_dict import SeedConfigTypedDict as SeedConfigTypedDict

SeedConfigInstanceOrDict = SeedConfig | SeedConfigTypedDict


__all__ = [
    "Config",
    "ConfigInstanceOrDict",
    "ConfigTypedDict",
    "CreateConfig",
    "CreateSeedConfig",
    "SeedConfig",
    "SeedConfigInstanceOrDict",
    "SeedConfigTypedDict",
]
