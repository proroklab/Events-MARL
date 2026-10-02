from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def save_checkpoint(path: str | Path, *, policy_params, critic_params, episode: int,
                    hn_params: Any = None, current_snd_ma: float | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "policy_params": policy_params,
        "critic_params": critic_params,
        "episode": episode,
    }
    if hn_params is not None:
        data["hn_params"] = hn_params
    if current_snd_ma is not None:
        data["current_snd_ma"] = current_snd_ma
    np.savez(path, **data)
    return path


def load_checkpoint(path: str | Path) -> dict:
    data = np.load(Path(path), allow_pickle=True)
    out = {}
    for key in data.files:
        value = data[key]
        out[key] = value.item() if value.dtype == object and value.shape == () else value
    return out

