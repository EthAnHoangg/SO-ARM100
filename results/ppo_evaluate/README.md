# PPO on fixed starts

Evaluation of PPO trained on the original fixed start (block always at `SUPPLY_POS`, arm at `READY_ARM_QPOS`, only
the colour varies). 300 episodes per mode on seeds 5000–5299, deterministic and stochastic, from
`shelfsort/evaluate.py`. `mean_of_colours` is the mean of the per-colour success rates.

## Dense (`ppo_dense_seed{0,1,2}.json`, models in `models/`)

- Setup: the v6 "dense 3M" configuration. 21-D observation, dense reward, PPO with SB3 defaults (64×64,
  lr 3e-4, `n_steps` 2048, batch 64, 10 epochs), 1 env, no `VecNormalize`, 3M steps, `--seed 0/1/2`.
- Seed 0 is the original v6 run (trained 2026-10-06 at `0707e85`). Seeds 1 and 2 were trained 2026-10-25 at
  `14e98a4`, whose `env.py` is identical to `0707e85`'s. All three were evaluated at `14e98a4`.
- These are 21-D models. Current `main` has the 27-D observation, so evaluate or play them from `14e98a4`.

| Seed | Det. | Stoch. | Stoch. R / G / B | Grasp (det.) |
|---|---|---|---|---|
| 0 | 1.00 | 0.79 | 0.82 / 0.81 / 0.75 | 1.00 |
| 1 | 0.00 | 0.02 | 0.05 / 0.01 / 0.00 | 1.00 |
| 2 | 1.00 | 0.90 | 0.88 / 0.92 / 0.91 | 1.00 |
| **Mean ± std** | **0.67 ± 0.58** | **0.57 ± 0.48** | | |

Seed 1 grasps and lifts the block in every episode but never releases it: it carries it slowly toward the zone with
the gripper shut until the 200-step limit. Its training success stayed at 0.02–0.13 from 1.25M steps to the end,
while its grasp rate was about 1.0 from 1M steps. The dense reward pays for grasping, lifting and carrying but nothing
pulls the policy toward releasing, so some seeds settle on grabbing and holding.

## Sparse (`ppo_sparse_seed{0,1,2}.json`)

Added in `0158f2c`. All three seeds score 0 deterministic and stochastic, with grasp 0.
