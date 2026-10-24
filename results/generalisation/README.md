# Randomised-reset generalisation results

Plan: `docs/superpowers/plans/2026-10-22-randomised-reset-generalisation.md`. Every number below comes from the
`eval.json` in each run folder (300 episodes, seeds 5000–5299; the radius sweep is deterministic only and has no arm or
zone noise). Each folder also has the final `model.zip`, its `vecnormalize.pkl` and the last 200 lines of the training
log. `play.py` and `eval_generalisation.py` both load `vecnormalize.pkl` automatically when given `<run>/model.zip`.

All runs: dense reward, PPO with SB3 defaults (64×64), 4 `SubprocVecEnv` envs, `VecNormalize`, 5M steps,
linear curriculum from scale 0 to 1 over the first 2M steps, 27-D observation.

## 1. Machine

- `nproc` = 4 (Intel Xeon Platinum 8259CL @ 2.50 GHz).
- E0 smoke run with default torch threading: **465 fps**. Torch's intra-op threads were competing with the 4 env
  subprocesses for the 4 cores. Setting `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` gave **791 fps** on the same smoke run,
  so every run below used it. This changes only thread counts, not the algorithm or hyperparameters.
- Measured on the 5M runs: 785–787 fps, about 1 h 46 min per run. With 4 cores the runs went one after another.

## 2. Baseline: v6 3M, trained on fixed starts

| Start condition | Det. | Stoch. | Grasp |
|---|---|---|---|
| L0 fixed | 1.00 | 0.79 | 1.00 |
| L1 block xy ±3 cm | 0.15 | 0.17 | 0.26 |
| L2 + arm joints (first 4) ±0.1 rad | 0.17 | 0.19 | 0.30 |
| L3 + zone xy ±2 cm | 0.16 | 0.17 | 0.31 |
| Block at fixed radius 1 / 2 / 3 / 4 / 5 cm (det.) | 0.44 / 0.17 / 0.10 / 0.07 / 0.06 | | grasp 0.84 / 0.28 / 0.13 / 0.05 / 0.05 |

## 3. Per-run results

The training levels are L2 for `dense_L2_*` and L3 for `dense_L3_*`. "Grasp" is the fraction of episodes in
which the block was ever held by both jaws. Steps to place counts successful episodes only.

#### `dense_L2_seed0`

| Start condition | Det. | Stoch. | Grasp (det.) | Det. R / G / B | Det. steps to place R / G / B |
|---|---|---|---|---|---|
| L0 | 0.67 | 0.65 | 1.00 | 1.00 / 1.00 / 0.00 | 40 / 64 / – |
| L1 | 0.67 | 0.66 | 1.00 | 1.00 / 1.00 / 0.00 | 42 / 65 / – |
| L2 | 0.65 | 0.65 | 1.00 | 0.98 / 0.97 / 0.00 | 45 / 65 / – |
| L3 | 0.53 | 0.52 | 1.00 | 0.75 / 0.83 / 0.01 | 52 / 70 / 155 |
| OOD_wide | 0.34 | 0.34 | 0.96 | 0.51 / 0.51 / 0.00 | 59 / 89 / – |

Radius sweep (det.): 0.67 / 0.67 / 0.66 / 0.64 / 0.51 for 1 / 2 / 3 / 4 / 5 cm; grasp 1.00 / 1.00 / 1.00 / 1.00 / 0.97.

#### `dense_L2_seed1`

| Start condition | Det. | Stoch. | Grasp (det.) | Det. R / G / B | Det. steps to place R / G / B |
|---|---|---|---|---|---|
| L0 | 1.00 | 0.99 | 1.00 | 1.00 / 1.00 / 1.00 | 38 / 50 / 61 |
| L1 | 1.00 | 0.99 | 1.00 | 1.00 / 0.99 / 1.00 | 41 / 54 / 62 |
| L2 | 0.99 | 0.99 | 1.00 | 0.99 / 1.00 / 0.97 | 42 / 54 / 64 |
| L3 | 0.86 | 0.89 | 0.99 | 0.89 / 0.80 / 0.90 | 45 / 62 / 65 |
| OOD_wide | 0.61 | 0.58 | 0.95 | 0.58 / 0.57 / 0.66 | 50 / 63 / 82 |

Radius sweep (det.): 1.00 / 1.00 / 1.00 / 0.94 / 0.86 for 1 / 2 / 3 / 4 / 5 cm; grasp 1.00 / 1.00 / 1.00 / 0.99 / 0.99.

#### `dense_L2_seed2`

| Start condition | Det. | Stoch. | Grasp (det.) | Det. R / G / B | Det. steps to place R / G / B |
|---|---|---|---|---|---|
| L0 | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 / 1.00 | 38 / 53 / 65 |
| L1 | 0.99 | 0.97 | 1.00 | 0.98 / 1.00 / 0.99 | 41 / 53 / 65 |
| L2 | 0.97 | 0.98 | 1.00 | 0.98 / 0.98 / 0.96 | 42 / 56 / 67 |
| L3 | 0.90 | 0.87 | 1.00 | 0.86 / 0.96 / 0.88 | 45 / 58 / 67 |
| OOD_wide | 0.64 | 0.64 | 0.97 | 0.67 / 0.67 / 0.59 | 61 / 66 / 72 |

Radius sweep (det.): 1.00 / 1.00 / 0.99 / 0.96 / 0.91 for 1 / 2 / 3 / 4 / 5 cm; grasp 1.00 / 1.00 / 1.00 / 0.99 / 0.99.

#### `dense_L3_seed0`

| Start condition | Det. | Stoch. | Grasp (det.) | Det. R / G / B | Det. steps to place R / G / B |
|---|---|---|---|---|---|
| L0 | 1.00 | 0.97 | 1.00 | 1.00 / 1.00 / 1.00 | 47 / 63 / 68 |
| L1 | 0.97 | 0.94 | 1.00 | 1.00 / 0.96 / 0.96 | 47 / 65 / 70 |
| L2 | 0.95 | 0.94 | 1.00 | 0.93 / 0.93 / 0.98 | 48 / 66 / 73 |
| L3 | 0.91 | 0.88 | 1.00 | 0.92 / 0.88 / 0.93 | 52 / 63 / 73 |
| OOD_wide | 0.74 | 0.72 | 0.96 | 0.74 / 0.71 / 0.76 | 58 / 72 / 82 |

Radius sweep (det.): 0.99 / 0.99 / 0.99 / 0.93 / 0.82 for 1 / 2 / 3 / 4 / 5 cm; grasp 1.00 / 1.00 / 1.00 / 1.00 / 0.95.

#### `dense_L3_seed1`

| Start condition | Det. | Stoch. | Grasp (det.) | Det. R / G / B | Det. steps to place R / G / B |
|---|---|---|---|---|---|
| L0 | 0.00 | 0.26 | 1.00 | 0.00 / 0.00 / 0.00 | – / – / – |
| L1 | 0.29 | 0.32 | 0.99 | 0.69 / 0.00 / 0.17 | 44 / – / 149 |
| L2 | 0.29 | 0.31 | 0.98 | 0.72 / 0.00 / 0.14 | 47 / – / 147 |
| L3 | 0.26 | 0.28 | 0.98 | 0.58 / 0.00 / 0.19 | 41 / – / 144 |
| OOD_wide | 0.21 | 0.25 | 0.89 | 0.42 / 0.03 / 0.18 | 43 / 164 / 140 |

Radius sweep (det.): 0.30 / 0.31 / 0.31 / 0.21 / 0.20 for 1 / 2 / 3 / 4 / 5 cm; grasp 1.00 / 1.00 / 0.99 / 0.97 / 0.80.

#### `dense_L3_seed2`

| Start condition | Det. | Stoch. | Grasp (det.) | Det. R / G / B | Det. steps to place R / G / B |
|---|---|---|---|---|---|
| L0 | 1.00 | 0.99 | 1.00 | 1.00 / 1.00 / 1.00 | 49 / 66 / 79 |
| L1 | 0.89 | 0.94 | 1.00 | 0.88 / 0.89 / 0.91 | 53 / 69 / 79 |
| L2 | 0.91 | 0.91 | 0.99 | 0.92 / 0.89 / 0.91 | 51 / 73 / 79 |
| L3 | 0.88 | 0.86 | 0.99 | 0.91 / 0.83 / 0.90 | 56 / 75 / 78 |
| OOD_wide | 0.64 | 0.60 | 0.96 | 0.63 / 0.67 / 0.61 | 66 / 82 / 86 |

Radius sweep (det.): 0.97 / 0.97 / 0.92 / 0.84 / 0.76 for 1 / 2 / 3 / 4 / 5 cm; grasp 1.00 / 1.00 / 0.99 / 0.98 / 0.97.


## 4. E2: three seeds trained on L2

Mean ± sample std (ddof = 1) of the per-colour-averaged success rate over seeds 0, 1 and 2.

| Level | Det. mean ± std | Stoch. mean ± std | Det. per-seed (0 / 1 / 2) |
|---|---|---|---|
| L0 | 0.89 ± 0.19 | 0.88 ± 0.20 | 0.67 / 1.00 / 1.00 |
| L1 | 0.88 ± 0.19 | 0.87 ± 0.19 | 0.67 / 1.00 / 0.99 |
| L2 | 0.87 ± 0.19 | 0.87 ± 0.19 | 0.65 / 0.99 / 0.97 |
| L3 | 0.76 ± 0.20 | 0.76 ± 0.20 | 0.53 / 0.86 / 0.90 |
| OOD_wide | 0.53 ± 0.17 | 0.52 ± 0.16 | 0.34 / 0.61 / 0.64 |
| radius_0.01 | 0.89 ± 0.19 | – | 0.67 / 1.00 / 1.00 |
| radius_0.02 | 0.89 ± 0.19 | – | 0.67 / 1.00 / 1.00 |
| radius_0.03 | 0.89 ± 0.19 | – | 0.66 / 1.00 / 0.99 |
| radius_0.04 | 0.85 ± 0.18 | – | 0.64 / 0.94 / 0.96 |
| radius_0.05 | 0.76 ± 0.22 | – | 0.51 / 0.86 / 0.91 |

**Decision gate after E1:** the L2 deterministic mean of `dense_L2_seed0` was 0.65 (≥ 0.6), so the plan went to the
seeds at 5M steps, with no 10M rerun. That figure passed the gate only because red and green were near 1.0; blue was
0.00 (see §6).

## 5. E3: three seeds trained on L3

Same statistics as §4, over `dense_L3_seed0`, `dense_L3_seed1` and `dense_L3_seed2`. Seed 1 collapsed (see §6), so
the std is large and the means sit well below the two healthy seeds.

| Level | Det. mean ± std | Stoch. mean ± std | Det. per-seed (0 / 1 / 2) |
|---|---|---|---|
| L0 | 0.67 ± 0.58 | 0.74 ± 0.41 | 1.00 / 0.00 / 1.00 |
| L1 | 0.72 ± 0.38 | 0.73 ± 0.36 | 0.97 / 0.29 / 0.89 |
| L2 | 0.71 ± 0.37 | 0.72 ± 0.35 | 0.95 / 0.29 / 0.91 |
| L3 | 0.68 ± 0.37 | 0.68 ± 0.34 | 0.91 / 0.26 / 0.88 |
| OOD_wide | 0.53 ± 0.28 | 0.52 ± 0.25 | 0.74 / 0.21 / 0.64 |
| radius_0.01 | 0.75 ± 0.40 | – | 0.99 / 0.30 / 0.97 |
| radius_0.02 | 0.75 ± 0.39 | – | 0.99 / 0.31 / 0.97 |
| radius_0.03 | 0.74 ± 0.38 | – | 0.99 / 0.31 / 0.92 |
| radius_0.04 | 0.66 ± 0.39 | – | 0.93 / 0.21 / 0.84 |
| radius_0.05 | 0.59 ± 0.34 | – | 0.82 / 0.20 / 0.76 |

## 6. What got better and what didn't

- **Grasping from randomised starts is fixed.** Every run grasps in 0.95–1.00 of episodes at L0–L3, against
  0.26–0.31 for v6. At a 5 cm block radius, v6 grasped 0.05 of the time; every new run except `dense_L3_seed1`
  (0.80) grasps at least 0.95.
- **Success at L1/L2 goes from about 0.16 to about 0.97–0.99** for the two seeds without a collapse (seeds 1 and 2),
  and they lose nothing at L0 (1.00 deterministic). Stochastic success at L0 rose from 0.79 (v6) to 0.99–1.00.
- **Seed 0 (L2) never places blue** (0.00 at every level, while grasp is 1.00). A diagnostic on 8 blue episodes
  (L0, deterministic) showed the same thing every time: the policy carries the block for 67 steps, then releases it
  at about (0.289, 0.085), 7.4 cm from the blue zone, and never gets closer. That is a local optimum that collects the
  transport reward for part of the way. Its red and green are about 1.0, so its mean is stuck at about 0.67. This
  outlier is what makes the three-seed std large (about 0.19). The plan says not to tune the reward or
  hyperparameters, so nothing was changed for it.
- **Zone jitter and wider starts are where generalisation falls off.** The L2-trained seeds drop to 0.86–0.90 on L3
  (zone jitter they never saw) and to 0.61–0.64 on OOD_wide. In the radius sweep, success falls past the 3 cm
  training range (0.94–0.96 at 4 cm, 0.86–0.91 at 5 cm) while grasp stays at 0.99, so the failures beyond the
  training range happen after the grasp: in the carry or the placement.
- **Training on L3 (E3)** doesn't clearly beat training on L2. The two healthy L3 seeds score 0.91 / 0.88 on L3
  (L2-trained healthy seeds: 0.86 / 0.90) and 0.74 / 0.64 on OOD_wide (L2: 0.61 / 0.64). They give up a little on
  L1/L2 (0.89–0.97) and fall off sooner in the radius sweep (0.82 / 0.76 at 5 cm). L3 seed 0 used seed 0 like the
  collapsed L2 run but placed every colour. With different ranges the same seed gives a different training run, so
  the collapse isn't fixed to seed 0.
- **`dense_L3_seed1` collapsed as well, on green.** It grasps 0.98–1.00 of the time at L0–L3 but never places green
  (0.00 at L0–L3, 0.03 on OOD_wide) and places blue only 0.14–0.19 of the time, taking about 145 steps when it does.
  The training log ends in the same state (success red 0.66, green 0, blue 0.22). At L0 its deterministic policy
  places nothing (0.00) while the stochastic one places red 0.67 of the time; this hasn't been diagnosed. That makes
  2 of the 6 runs stuck on a colour, so on this setup a collapse is a regular outcome, not a one-off. Nothing was
  changed for it, for the same reason as above.
- **Weakest colour:** blue is the slowest to place in every run (most steps to place, because it's the farthest
  zone), and L2 seed 0 never places it at all. L3 seed 1 never places green. Apart from the two collapsed runs, no
  colour is consistently the weakest: green is weakest for L2 seed 1 on L3 (0.80) and L3 seed 2 on L3 (0.83), and
  blue for L2 seed 2 on OOD_wide (0.59).

## 7. Task 3 STOP condition

Nothing to report: the scripted pinch grasped and lifted the block from all 12 randomised starts (4 corners at
±3 cm plus 8 random L2 starts), so no test is marked `xfail`.

## Notes

- A bug found while writing this up and fixed in `a49494b`: `load_policy` looked only for
  `<stem>_vecnormalize.pkl`, so the committed `<run>/model.zip` would have been run on raw observations with only a
  warning. It now falls back to `vecnormalize.pkl` in the same folder. All the evaluations above ran on
  `runs/gen/models/<run>.zip`, which has its stem-named stats file, so they weren't affected.
- A second bug, fixed after every run above was trained and evaluated: `reset()` did not clip the arm noise to the
  joint limits. `wrist_flex`'s ready angle (1.637) sits 0.021 rad under its limit (1.658), so ±0.1 rad noise started it
  past the limit in about 37% of L2/L3 episodes (by up to 0.078 rad). The joint-limit constraint pushed it back inside
  within about 4 steps, a transient about the size of the normal start-up settling, so the effect on these results is
  likely small. But the first observation in those episodes was a pose the real arm cannot reach, and `wrist_flex`'s
  effective range was about −0.1 to +0.021 rad rather than ±0.1. All six runs and their evaluations used the unclipped
  starts; runs trained after the fix (including any SAC runs) use clipped starts, so they are not on exactly the same
  start distribution.
