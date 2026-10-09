from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Checkpoints written before the NMD/SND rename stored either running average under this key.
_LEGACY_EMA_KEY = "current_snd_ma"


def save_checkpoint(path: str | Path, *, policy_params, critic_params, episode: int,
                    hn_params: Any = None, nmd_ema: float | None = None,
                    snd_ema: float | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "policy_params": policy_params,
        "critic_params": critic_params,
        "episode": episode,
    }
    if hn_params is not None:
        data["hn_params"] = hn_params
    if nmd_ema is not None:
        data["nmd_ema"] = nmd_ema
    if snd_ema is not None:
        data["snd_ema"] = snd_ema
    np.savez(path, **data)
    return path


def load_checkpoint(path: str | Path) -> dict:
    data = np.load(Path(path), allow_pickle=True)
    out = {}
    for key in data.files:
        value = data[key]
        out[key] = value.item() if value.dtype == object and value.shape == () else value

    legacy = out.pop(_LEGACY_EMA_KEY, None)
    if legacy is not None:
        # Only Ours has a hypernetwork, so its old checkpoints held the NMD average; DiCo's held
        # its SND average. PS wrote the key too, but nothing reads a diversity average for PS.
        if "hn_params" in out:
            out.setdefault("nmd_ema", legacy)
        elif "hetero_network" in out["policy_params"]:
            out.setdefault("snd_ema", legacy)
    return out
