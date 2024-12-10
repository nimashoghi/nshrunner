from __future__ import annotations

__codegen__ = True

from nshrunner import Config as Config
from nshrunner import RunnerConfig as RunnerConfig
from nshrunner._seed import SeedConfig as SeedConfig
from nshrunner.backends.screen import LoggingConfig as LoggingConfig
from nshrunner.backends.screen import ScreenBackendConfig as ScreenBackendConfig
from nshrunner.backends.slurm import SlurmBackendConfig as SlurmBackendConfig
from nshrunner.backends.slurm import SlurmMailConfig as SlurmMailConfig

from .Config_typed_dict import ConfigTypedDict as ConfigTypedDict
from .Config_typed_dict import CreateConfig as CreateConfig

ConfigInstanceOrDict = Config | ConfigTypedDict

from .backends.screen.LoggingConfig_typed_dict import (
    CreateLoggingConfig as CreateLoggingConfig,
)
from .backends.screen.LoggingConfig_typed_dict import (
    LoggingConfigTypedDict as LoggingConfigTypedDict,
)

LoggingConfigInstanceOrDict = LoggingConfig | LoggingConfigTypedDict

from .backends.screen.ScreenBackendConfig_typed_dict import (
    CreateScreenBackendConfig as CreateScreenBackendConfig,
)
from .backends.screen.ScreenBackendConfig_typed_dict import (
    ScreenBackendConfigTypedDict as ScreenBackendConfigTypedDict,
)

ScreenBackendConfigInstanceOrDict = ScreenBackendConfig | ScreenBackendConfigTypedDict

from ._seed.SeedConfig_typed_dict import CreateSeedConfig as CreateSeedConfig
from ._seed.SeedConfig_typed_dict import SeedConfigTypedDict as SeedConfigTypedDict

SeedConfigInstanceOrDict = SeedConfig | SeedConfigTypedDict

from .backends.slurm.SlurmBackendConfig_typed_dict import (
    CreateSlurmBackendConfig as CreateSlurmBackendConfig,
)
from .backends.slurm.SlurmBackendConfig_typed_dict import (
    SlurmBackendConfigTypedDict as SlurmBackendConfigTypedDict,
)

SlurmBackendConfigInstanceOrDict = SlurmBackendConfig | SlurmBackendConfigTypedDict

from .backends.slurm.SlurmMailConfig_typed_dict import (
    CreateSlurmMailConfig as CreateSlurmMailConfig,
)
from .backends.slurm.SlurmMailConfig_typed_dict import (
    SlurmMailConfigTypedDict as SlurmMailConfigTypedDict,
)

SlurmMailConfigInstanceOrDict = SlurmMailConfig | SlurmMailConfigTypedDict


from . import _runner as _runner
from . import _seed as _seed
from . import backends as backends

__all__ = [
    "Config",
    "ConfigInstanceOrDict",
    "ConfigTypedDict",
    "CreateConfig",
    "CreateLoggingConfig",
    "CreateScreenBackendConfig",
    "CreateSeedConfig",
    "CreateSlurmBackendConfig",
    "CreateSlurmMailConfig",
    "LoggingConfig",
    "LoggingConfigInstanceOrDict",
    "LoggingConfigTypedDict",
    "RunnerConfig",
    "ScreenBackendConfig",
    "ScreenBackendConfigInstanceOrDict",
    "ScreenBackendConfigTypedDict",
    "SeedConfig",
    "SeedConfigInstanceOrDict",
    "SeedConfigTypedDict",
    "SlurmBackendConfig",
    "SlurmBackendConfigInstanceOrDict",
    "SlurmBackendConfigTypedDict",
    "SlurmMailConfig",
    "SlurmMailConfigInstanceOrDict",
    "SlurmMailConfigTypedDict",
    "_runner",
    "_seed",
    "backends",
]
