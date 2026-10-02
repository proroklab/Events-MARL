from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _plain(value: Any):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    array = np.asarray(value)
    return array.item() if array.size == 1 else array.tolist()


class RunLogger:
    def __init__(self, log_dir: str | Path, csv_fields=("episode", "avg_reward", "policy_loss",
                                                        "value_loss", "entropy", "nmd", "alpha")):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.dir / "metrics.jsonl"
        self.csv = self.dir / "metrics.csv"
        self.csv_fields = csv_fields
        if not self.csv.exists():
            self.csv.write_text(",".join(csv_fields) + "\n")

    def log(self, record: Mapping[str, Any]) -> None:
        row = {k: _plain(v) for k, v in record.items()}
        with self.jsonl.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        with self.csv.open("a") as handle:
            handle.write(",".join(str(row.get(f, "")) for f in self.csv_fields) + "\n")

