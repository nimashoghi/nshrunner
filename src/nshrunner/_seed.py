from __future__ import annotations

import logging
import os

import nshconfig as C

try:
    import lightning.fabric.utilities.seed as LS  # pyright: ignore[reportMissingImports]
except ImportError:
    LS = None

log = logging.getLogger(__name__)


class SeedConfig(C.Config):
    seed: int
    """Seed for the random number generator."""

    seed_workers: bool = False
    """Whether to seed the workers of the dataloader (Only applicable to PyTorch Lightning)."""

    use_lightning: bool = True
    """Whether to use Lightning's seed_everything function (if available)."""


def seed_everything(config: SeedConfig):
    # If Lightning's seed_everything is not available, we just use own implementation
    if LS is not None:
        seed = LS.seed_everything(config.seed, workers=config.seed_workers)
        log.critical(f"Set global seed to {config.seed}.")
        return seed

    # First, set `random` seed
    import random

    random.seed(config.seed)

    # Then, set `numpy` seed
    try:
        import numpy as np  # pyright: ignore[reportMissingImports]

        np.random.seed(config.seed)
    except ImportError:
        pass

    # Finally, set `torch` seed
    try:
        import torch  # pyright: ignore[reportMissingImports]

        torch.manual_seed(config.seed)
    except ImportError:
        pass

    os.environ["PL_GLOBAL_SEED"] = str(config.seed)
    os.environ["PL_SEED_WORKERS"] = str(int(config.seed_workers))

    log.critical(f"Set global seed to {config.seed}.")
    return config.seed
