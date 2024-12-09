from __future__ import annotations

__codegen__ = True

from nshrunner.wrapper_fns import Config as Config

from .Config_typed_dict import ConfigTypedDict as ConfigTypedDict
from .Config_typed_dict import CreateConfig as CreateConfig

__all__ = [
    "Config",
    "ConfigTypedDict",
    "CreateConfig",
]
