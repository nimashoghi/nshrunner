import logging

try:
    import lightning.fabric.utilities.seed as LS  # pyright: ignore[reportMissingImports]
except ImportError:
    LS = None

log = logging.getLogger(__name__)


def seed_everything(seed: int, *, workers: bool = False):
    # If Lightning's seed_everything is not available, we just use own implementation
    if LS is not None:
        return LS.seed_everything(seed, workers=workers)

    # First, set `random` seed
    import random

    random.seed(seed)

    # Then, set `numpy` seed
    try:
        import numpy as np  # pyright: ignore[reportMissingImports]

        np.random.seed(seed)
    except ImportError:
        pass

    # Finally, set `torch` seed
    try:
        import torch  # pyright: ignore[reportMissingImports]

        torch.manual_seed(seed)
    except ImportError:
        pass

    log.critical(f"Set global seed to {seed}.")
