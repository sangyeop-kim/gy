# RL DQN Training

This document describes how to train, evaluate, and compare the DQN dispatch policy.

## Environment

Install dependencies:

```bash
uv sync
```

Check the PyTorch device:

```bash
python -c "import torch; print(torch.__version__); print('cuda', torch.cuda.is_available()); print('mps', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())"
```

The training CLI uses `--device auto` by default. It selects `cuda` first, then `mps`, then `cpu`.

## Quick Smoke Test

Use this to verify the full RL path without spending much time:

```bash
python -m src.rl.train_dqn \
  --episodes 1 \
  --max-lots 200 \
  --eval-max-lots 200 \
  --output-dir outputs/rl_dqn_smoke \
  --batch-size 64 \
  --hidden-dim 64 \
  --hidden-layers 2 \
  --device auto
```

Expected outputs:

- `outputs/rl_dqn_smoke/dqn_agent.pkl`
- `outputs/rl_dqn_smoke/training_episodes.csv`
- `outputs/rl_dqn_smoke/evaluation.csv`
- `outputs/rl_dqn_smoke/feature_names.json`

Small runs can block because batch minimum constraints cannot be satisfied by the last waiting lots. This is expected under the current batching model.

## Recommended 1000-Lot Training

Start with this setting for an initial experiment:

```bash
python -m src.rl.train_dqn \
  --episodes 10 \
  --max-lots 1000 \
  --eval-max-lots 1000 \
  --output-dir outputs/rl_dqn_1000 \
  --seed 42 \
  --hidden-dim 128 \
  --hidden-layers 2 \
  --learning-rate 3e-4 \
  --batch-size 256 \
  --epsilon-start 1.0 \
  --epsilon-end 0.05 \
  --epsilon-decay 0.9995 \
  --fallback-policy priority_cr_fifo \
  --fallback-probability 0.05 \
  --progress \
  --progress-interval-events 50000 \
  --device auto
```

For a longer run:

```bash
python -m src.rl.train_dqn \
  --episodes 30 \
  --max-lots 1000 \
  --eval-max-lots 1000 \
  --output-dir outputs/rl_dqn_1000_long \
  --seed 42 \
  --hidden-dim 256 \
  --hidden-layers 3 \
  --learning-rate 2e-4 \
  --batch-size 512 \
  --epsilon-start 1.0 \
  --epsilon-end 0.03 \
  --epsilon-decay 0.9997 \
  --fallback-probability 0.05 \
  --progress \
  --progress-interval-events 50000 \
  --device auto
```

## Hyperparameters

Core parameters:

- `--episodes`: number of repeated simulator episodes used for training.
- `--max-lots`: lots loaded per training episode.
- `--eval-max-lots`: lots used in the final greedy evaluation after training.
- `--hidden-dim`: DQN hidden width.
- `--hidden-layers`: number of hidden layers.
- `--learning-rate`: AdamW learning rate.
- `--batch-size`: replay minibatch size.
- `--epsilon-start`: initial random exploration probability.
- `--epsilon-end`: minimum exploration probability.
- `--epsilon-decay`: multiplicative decay applied after each training step.
- `--fallback-policy`: simulator fallback dispatch rule. This is also used as a weak imitation fallback when `--fallback-probability > 0`.
- `--fallback-probability`: probability of using the fallback rule during training decisions.
- `--progress` / `--no-progress`: print simulator and DQN progress during each episode.
- `--progress-interval-events`: simulator event interval between progress prints.
- `--device`: `auto`, `cpu`, `cuda`, `cuda:0`, or `mps`.

Practical tuning notes:

- If training is noisy, lower `--learning-rate` to `1e-4` or increase `--batch-size`.
- If the agent collapses too early, increase `--epsilon-decay` closer to `1.0`.
- If learning is too random for too long, lower `--epsilon-decay`, for example `0.999`.
- Use `--fallback-probability 0.05` to keep occasional high-quality dispatch examples from `priority_cr_fifo`.
- For GPU training, larger `--batch-size` such as `512` is reasonable.

## State, Action, Reward

Action:

- One dispatch action selects one waiting lot from the current candidate set.
- For batch steps, the selected lot is the lead lot; compatible lots are then batched by simulator base logic.

State features:

- current simulation time
- queue size
- idle/busy/down tool ratios
- lot priority and super-hot flag
- waiting age and release age
- due date slack
- critical ratio
- current step processing time
- remaining route processing time
- route progress
- wafers per lot
- setup preview
- batch step flag
- batch min/max
- compatible batch quantity
- batch minimum satisfied flag
- active CQT remaining time

Reward:

- immediate dense reward for selecting urgent low-slack/low-CR lots
- immediate penalty for selecting a lot with avoidable setup time
- immediate reward for selecting batch leads whose compatible quantity satisfies batch minimum
- immediate penalty for selecting batch leads that cannot form a valid batch
- immediate reward for selecting lots inside tight active CQT windows
- positive reward for remaining processing reduction
- positive reward for projected tardiness improvement
- on-time lot completion bonus
- CQT pass bonus
- lot completion bonus
- penalty for projected tardiness
- penalty for late completion
- penalty for CQT violations and CQT window consumption
- cycle time penalty
- waiting age penalty
- priority and super-hot lots increase due-date penalty weight

Reward component averages are written into `training_episodes.csv` and `evaluation.csv` with `reward_component_*` columns.

## Run a Trained Checkpoint

After training, run the simulator using the saved DQN checkpoint as a dispatch rule:

```bash
python -m src.rl.simulate_dqn \
  --checkpoint outputs/rl_dqn_1000/dqn_agent.pkl \
  --max-lots 1000 \
  --seed 42 \
  --output-dir outputs/rl_dqn_1000_sim \
  --device auto
```

This runs with:

- `explore=False`
- `train_online=False`
- fixed checkpoint weights

Outputs:

- `summary.json`
- `summary.csv`
- `snapshot_summary.csv`
- `lots.csv`
- `product_summary.csv`
- `toolgroup_summary.csv`
- `tool_summary.csv`
- `blocked_waiting_lots.csv` if blocked
- `blocked_waiting_steps.csv` if blocked

Use `--write-event-log` only when event-level logs are needed. Event logs can be large.

## Compare Against Existing Rules

Run the baseline dispatch rules:

```bash
python -m src.compare_policies \
  --max-lots 1000 \
  --seed 42 \
  --output-dir outputs/policy_comparison_1000 \
  --no-save-events
```

Run the trained RL checkpoint:

```bash
python -m src.rl.simulate_dqn \
  --checkpoint outputs/rl_dqn_1000/dqn_agent.pkl \
  --max-lots 1000 \
  --seed 42 \
  --output-dir outputs/rl_dqn_1000_sim \
  --device auto
```

Combine the summary tables:

```bash
python - <<'PY'
import pandas as pd

baseline = pd.read_csv("outputs/policy_comparison_1000/overall_policy_summary.csv")
rl = pd.read_csv("outputs/rl_dqn_1000_sim/summary.csv")
combined = pd.concat([baseline, rl], ignore_index=True, sort=False)
combined.to_csv("outputs/combined_policy_summary_1000.csv", index=False)
print(combined[[
    "policy",
    "completed_lots",
    "released_lots",
    "completed_ratio",
    "average_cycle_time_minutes",
    "average_tardiness_minutes",
    "tardy_lots",
    "on_time_ratio",
    "cqt_violations",
    "blocked_reason",
]].to_string(index=False))
PY
```

This gives one table containing:

- `fifo`
- `spt`
- `edd`
- `critical_ratio`
- `priority_cr_fifo`
- `rl_dqn`

## Notes

- RL and rule-based runs use the same simulator base logic. The only intended difference is dispatch lot selection.
- `simulate_dqn` uses greedy checkpoint inference, so it does not keep learning during simulation.
- `fallback-policy` in `simulate_dqn` is plumbing fallback only; dispatch decisions are made by the checkpoint.
- Current small-lot experiments often end with `waiting_lots_cannot_satisfy_dispatch_constraints` because strict batch minimum constraints leave residual lots that cannot form a batch.
