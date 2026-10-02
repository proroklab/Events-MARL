# Events as Triggers for Behavioral Diversity in Multi-Agent Reinforcement Learning

Code to train the methods compared in *Events as Triggers for Behavioral Diversity in Multi-Agent
Reinforcement Learning* — Ours, parameter sharing (PS), DiCo, CASH and HyperMARL — on the paper's
environments.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

`requirements.txt` pins the versions the package was developed with (JAX 0.10.2, CPU). JAX changes
floating-point behaviour between releases, so other versions give similar but not identical runs.

## Train

```bash
python -m hyperlora.cli.train --config configs/main/ours/navigation.yaml --seed 42 \
    --out runs/ours_navigation_seed42
```

Any config value can be overridden with `--set section.key=value`, e.g.
`--set ppo.num_episodes=500`. Runs are deterministic for a given config, seed and runtime.

HyperMARL has its own trainer and config format:

```bash
python -m hyperlora.cli.train_hypermarl --config configs/main/hypermarl/navigation.yaml --seed 42 \
    --out runs/hypermarl_navigation_seed42
```

Its overrides are top-level keys: `--set TOTAL_TIMESTEPS=1000000`.

The published runs used seeds 0, 42, 342, 3421 and 34210.

## Configs

One fully specified config per method and environment in `configs/main/`. Every key is required;
unknown keys are rejected.

| Environment | Ours, PS, DiCo, CASH | HyperMARL |
|---|---|---|
| Navigation | `configs/main/<method>/navigation.yaml` | `configs/main/hypermarl/navigation.yaml` |
| Dispersion | `configs/main/<method>/dispersion.yaml` | `configs/main/hypermarl/dispersion.yaml` |
| Reverse Transport | `configs/main/<method>/reverse_transport.yaml` | `configs/main/hypermarl/reverse_transport.yaml` |
| Pressure Plate | `configs/main/<method>/pressure_plate.yaml` | `configs/main/hypermarl/pressure_plate.yaml` |
| Football | `configs/main/<method>/football.yaml` | `configs/main/hypermarl/football.yaml` |
| Wind Flocking | `configs/main/ours/wind_flocking.yaml` (Ours only) | — |

## Output

`--out runs/<name>` writes:

* `logs/metrics.jsonl` — one row per training episode (reward, losses, policy std, diversity
  measures) and one row per evaluation, marked `"eval": true`, with `completion_rate` in percent and
  `avg_agents_at_goal_percentage`; on Navigation also `success_rate`, the percentage of food found;
* `logs/resolved_config.yaml` — the config the run actually used, overrides included;
* `checkpoints/checkpoint_<episode>.npz` every `logging.save_interval` episodes, and
  `checkpoints/final_checkpoint.npz`.

HyperMARL writes `metrics.jsonl`, `eval_metrics.json`, `final_params.npz` and `resolved_config.json`
into its `--out` directory; its evaluation rows carry `eval/completion_rate`, and on Navigation
`eval/success_rate`.

## Layout

```
configs/main/<method>/<env>.yaml   one config per method and environment
src/hyperlora/
  config.py                        the typed config and its loader
  cli/                             train.py, train_hypermarl.py
  envs/                            environment registry, factory, capabilities, hypernetwork tokens,
                                   event triggers
  envs/scenarios/                  the VMAS scenarios
  models/                          policies, hypernetwork, critic
  algorithms/                      the PPO update of each method: ours, ps, dico, cash
  training/                        rollout, GAE, minibatching, checkpointing, the training loops
  diversity/                       NMD (Ours), SND (DiCo), the diversity control law
  evaluation/                      the in-training evaluation (completion rate)
  baselines/hypermarl/             the HyperMARL trainer and its adapter
```
