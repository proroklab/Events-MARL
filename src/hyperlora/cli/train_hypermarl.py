from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import yaml


METRIC_RENAMES = {
    "dispersion": {"food_found_percentage": "success_rate",
                   "eval/food_found_percentage": "eval/success_rate"},
}


class _RunLogger:
    def __init__(self, path: Path, renames: dict | None = None):
        self.dir = str(path.parent)
        self._renames = renames or {}
        self._path = path
        self.rows: list = []
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    def log(self, data, step=None, **_kwargs):
        row = {"step": None if step is None else int(step),
               **{self._renames.get(k, k): _plain(v) for k, v in data.items()}}
        self.rows.append(row)
        with self._path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    def summary(self, *_args, **_kwargs):
        pass


def _plain(value):
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else array.tolist()


def train(config: dict, out_dir: Path, overrides: dict | None = None):
    import jax

    from ..baselines.hypermarl.compat import install

    hypermarl = install()
    config = dict(config)
    config.update(overrides or {})

    threefry = config.pop("JAX_THREEFRY_PARTITIONABLE")
    if not isinstance(threefry, bool):
        raise ValueError("JAX_THREEFRY_PARTITIONABLE must be true or false")
    jax.config.update("jax_threefry_partitionable", threefry)

    env_config = config["env"]["env"] if "env" in config["env"] else config["env"]
    config["TEST_ENV_KWARGS"].update(config["env"]["ENV_KWARGS"])
    config["TRAIN_ENV_KWARGS"].update(config["env"]["ENV_KWARGS"])
    for key in ("use_fixed_capabilities", "fixed_capabilities"):
        if key in env_config:
            config["TRAIN_ENV_KWARGS"][key] = env_config[key]
            config["TEST_ENV_KWARGS"][key] = env_config[key]
    config.setdefault("CHP_DIR", str(out_dir / "checkpoints"))

    renames = METRIC_RENAMES.get(config["ENV_NAME"], {})
    logger = _RunLogger(out_dir / "metrics.jsonl", renames)
    out, _env = hypermarl.make_train(config)(jax.random.PRNGKey(config["SEED"]), logger)

    flat: dict = {}

    def walk(node, prefix):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{prefix}/{key}")
        else:
            flat[prefix] = np.asarray(node)

    walk(out["runner_state"][0].params, "params")
    np.savez_compressed(out_dir / "final_params.npz", **flat)
    np.savez_compressed(out_dir / "updates.npz",
                        **{k: np.asarray(v) for k, v in out["metrics"].items()})
    (out_dir / "eval_metrics.json").write_text(json.dumps(
        [{"step": int(step), **{renames.get(k, k): float(np.asarray(v)) for k, v in metrics.items()}}
         for step, metrics in out["eval_metrics"]], indent=1))
    (out_dir / "resolved_config.json").write_text(json.dumps(
        {**config, "JAX_THREEFRY_PARTITIONABLE": threefry}, indent=1, default=str))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="override a top-level config key, e.g. --set NUM_ENVS=8")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    if args.seed is not None:
        config["SEED"] = args.seed
    for item in args.overrides:
        key, _, raw = item.partition("=")
        try:
            config[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            config[key] = raw
    out_dir = args.out or Path("runs") / f"hypermarl_{args.config.stem}_seed{config['SEED']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    from ..seeding import seed_before_vmas_import

    seed_before_vmas_import(config["SEED"])
    train(config, out_dir)
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()

