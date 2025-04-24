from __future__ import annotations

__codegen__ = True

from nshrunner._runner import Config as Config

from .Config_typed_dict import ConfigTypedDict as ConfigTypedDict
from .Config_typed_dict import CreateConfig as CreateConfig

ConfigInstanceOrDict = Config | ConfigTypedDict


__all__ = [
    "Config",
    "ConfigInstanceOrDict",
    "ConfigTypedDict",
    "CreateConfig",
]
