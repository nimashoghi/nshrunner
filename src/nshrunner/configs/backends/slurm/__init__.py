from __future__ import annotations

__codegen__ = True

from nshrunner.backends.slurm import SlurmBackendConfig as SlurmBackendConfig
from nshrunner.backends.slurm import SlurmMailConfig as SlurmMailConfig

from .SlurmBackendConfig_typed_dict import (
    CreateSlurmBackendConfig as CreateSlurmBackendConfig,
)
from .SlurmBackendConfig_typed_dict import (
    SlurmBackendConfigTypedDict as SlurmBackendConfigTypedDict,
)

SlurmBackendConfigInstanceOrDict = SlurmBackendConfig | SlurmBackendConfigTypedDict

from .SlurmMailConfig_typed_dict import CreateSlurmMailConfig as CreateSlurmMailConfig
from .SlurmMailConfig_typed_dict import (
    SlurmMailConfigTypedDict as SlurmMailConfigTypedDict,
)

SlurmMailConfigInstanceOrDict = SlurmMailConfig | SlurmMailConfigTypedDict


__all__ = [
    "CreateSlurmBackendConfig",
    "CreateSlurmMailConfig",
    "SlurmBackendConfig",
    "SlurmBackendConfigInstanceOrDict",
    "SlurmBackendConfigTypedDict",
    "SlurmMailConfig",
    "SlurmMailConfigInstanceOrDict",
    "SlurmMailConfigTypedDict",
]
