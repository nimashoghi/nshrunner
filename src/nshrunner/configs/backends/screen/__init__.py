from __future__ import annotations

__codegen__ = True

from nshrunner.backends.screen import LoggingConfig as LoggingConfig
from nshrunner.backends.screen import ScreenBackendConfig as ScreenBackendConfig

from .LoggingConfig_typed_dict import CreateLoggingConfig as CreateLoggingConfig
from .LoggingConfig_typed_dict import LoggingConfigTypedDict as LoggingConfigTypedDict

LoggingConfigInstanceOrDict = LoggingConfig | LoggingConfigTypedDict

from .ScreenBackendConfig_typed_dict import (
    CreateScreenBackendConfig as CreateScreenBackendConfig,
)
from .ScreenBackendConfig_typed_dict import (
    ScreenBackendConfigTypedDict as ScreenBackendConfigTypedDict,
)

ScreenBackendConfigInstanceOrDict = ScreenBackendConfig | ScreenBackendConfigTypedDict


__all__ = [
    "CreateLoggingConfig",
    "CreateScreenBackendConfig",
    "LoggingConfig",
    "LoggingConfigInstanceOrDict",
    "LoggingConfigTypedDict",
    "ScreenBackendConfig",
    "ScreenBackendConfigInstanceOrDict",
    "ScreenBackendConfigTypedDict",
]
