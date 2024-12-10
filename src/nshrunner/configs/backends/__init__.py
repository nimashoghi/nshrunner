from __future__ import annotations

__codegen__ = True

from nshrunner.backends.screen import LoggingConfig as LoggingConfig
from nshrunner.backends.screen import ScreenBackendConfig as ScreenBackendConfig
from nshrunner.backends.slurm import SlurmBackendConfig as SlurmBackendConfig
from nshrunner.backends.slurm import SlurmMailConfig as SlurmMailConfig

from .screen.LoggingConfig_typed_dict import CreateLoggingConfig as CreateLoggingConfig
from .screen.LoggingConfig_typed_dict import (
    LoggingConfigTypedDict as LoggingConfigTypedDict,
)

LoggingConfigInstanceOrDict = LoggingConfig | LoggingConfigTypedDict

from .screen.ScreenBackendConfig_typed_dict import (
    CreateScreenBackendConfig as CreateScreenBackendConfig,
)
from .screen.ScreenBackendConfig_typed_dict import (
    ScreenBackendConfigTypedDict as ScreenBackendConfigTypedDict,
)

ScreenBackendConfigInstanceOrDict = ScreenBackendConfig | ScreenBackendConfigTypedDict

from .slurm.SlurmBackendConfig_typed_dict import (
    CreateSlurmBackendConfig as CreateSlurmBackendConfig,
)
from .slurm.SlurmBackendConfig_typed_dict import (
    SlurmBackendConfigTypedDict as SlurmBackendConfigTypedDict,
)

SlurmBackendConfigInstanceOrDict = SlurmBackendConfig | SlurmBackendConfigTypedDict

from .slurm.SlurmMailConfig_typed_dict import (
    CreateSlurmMailConfig as CreateSlurmMailConfig,
)
from .slurm.SlurmMailConfig_typed_dict import (
    SlurmMailConfigTypedDict as SlurmMailConfigTypedDict,
)

SlurmMailConfigInstanceOrDict = SlurmMailConfig | SlurmMailConfigTypedDict


from . import screen as screen
from . import slurm as slurm

__all__ = [
    "CreateLoggingConfig",
    "CreateScreenBackendConfig",
    "CreateSlurmBackendConfig",
    "CreateSlurmMailConfig",
    "LoggingConfig",
    "LoggingConfigInstanceOrDict",
    "LoggingConfigTypedDict",
    "ScreenBackendConfig",
    "ScreenBackendConfigInstanceOrDict",
    "ScreenBackendConfigTypedDict",
    "SlurmBackendConfig",
    "SlurmBackendConfigInstanceOrDict",
    "SlurmBackendConfigTypedDict",
    "SlurmMailConfig",
    "SlurmMailConfigInstanceOrDict",
    "SlurmMailConfigTypedDict",
    "screen",
    "slurm",
]
