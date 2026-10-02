from __future__ import annotations

import random

import numpy as np


def seed_before_vmas_import(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def seed_globals(seed: int) -> np.random.Generator:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return np.random.default_rng(seed)

