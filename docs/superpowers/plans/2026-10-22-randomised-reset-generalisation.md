# Randomised-Reset Generalisation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a dense-reward PPO policy for ShelfSort that still places blocks when the block, arm and zone start positions are randomised, and measure how well it generalises.

**Architecture:** `ShelfSortEnv` gets optional per-episode randomisation of block xy, arm joints and zone xy, scaled by a curriculum factor. The observation grows from 21-D to 27-D by appending `block − ee` and `target − block`. `train.py` gains vectorised envs, `VecNormalize`, a linear curriculum callback and checkpoints. A new `shelfsort/eval_generalisation.py` scores a model on fixed, randomised and out-of-distribution starts. Then the experiments in Task 6 run, and the results are committed under `results/generalisation/`.

**Tech Stack:** Python 3.11, uv, MuJoCo 3.10, Gymnasium ≥ 1.0, Stable-Baselines3 2.4.0 (PPO), PyTorch 2.2.2 (CPU), pytest.

**Spec:** This plan has no separate spec. The *Background* section below is the spec.

## Background (why this exists)

The current best policy is dense PPO "v6 3M": 64×64 network, SB3 defaults, 1 env, seed 0, 3M steps. Every episode starts the same way: same arm pose, block always at `SUPPLY_POS = (0.16, −0.18)`, and only the colour (red/green/blue → zone) varies. On 300 eval episodes (seeds 5000–5299) it scores 1.00 deterministic and 0.79 stochastic.

A generalisation check on 2026-10-22 ran that model on randomised starts (same seeds) and it failed badly:

| Start condition | Det. | Stoch. | Grasp |
|---|---|---|---|
| L0 fixed | 1.00 | 0.79 | 1.00 |
| L1 block xy ±3 cm | 0.15 | 0.17 | 0.26 |
| L2 + arm joints (first 4) ±0.1 rad | 0.17 | 0.19 | 0.30 |
| L3 + zone xy ±2 cm | 0.16 | 0.17 | 0.31 |
| Block at fixed radius 1 / 2 / 3 / 4 / 5 cm (det.) | 0.44 / 0.17 / 0.10 / 0.07 / 0.06 | | grasp 0.84 / 0.28 / 0.13 / 0.05 / 0.05 |

The policy memorised where the block usually is, so it fails at the grasp. The fix is to train on randomised starts, give the policy relative position vectors, and normalise the observations. Those v6 numbers are the baseline every result here is compared against. The v6 model itself is not in the repo (`runs/` is gitignored), so do not try to re-run it.

## Global Constraints

- Use **uv** for everything: `uv sync`, `uv run pytest`, `uv run python -m ...`. Never use pip or venv. Commit `uv.lock` if it changes.
- Do not change pinned versions (`mujoco==3.10`, `stable-baselines3==2.4.0`, `torch==2.2.2`). Python is pinned to 3.11 in `.python-version`.
- Do not change the reward terms or their weights (`REACH_WEIGHT`, `TRANSPORT_WEIGHT`, `GRASP_BONUS`, `LIFT_WEIGHT`, `LIFT_CAP`, `LOWER_RAMP`, `STEP_PENALTY`, `PLACEMENT_REWARD`) or the success rule. The only experimental variables are randomisation, observation and normalisation.
- Do not modify `Simulation/SO101/*.xml`.
- With all randomisation ranges at 0, reset must behave exactly as it does today: block at `SUPPLY_POS`, arm at `READY_ARM_QPOS`, zones at `SHELF_ZONE_POS`.
- PPO: `MlpPolicy` with SB3 defaults (64×64, lr 3e-4, `n_steps=2048` per env, batch 64, 10 epochs, γ 0.99, λ 0.95, clip 0.2, ent 0), `device="cpu"`, **4 envs** (`SubprocVecEnv`) for every training run in Task 6, whatever the machine's core count.
- Evaluation always uses seeds 5000–5299 (300 episodes) and reports per-colour success, grasp rate and mean steps to place, deterministic and stochastic.
- Work on branch `feat/randomised-reset`. Never push to `main` or open a PR to `TheRobotStudio/SO-ARM100` (`origin`). Push only to `fork`.
- Comment density and style should match `shelfsort/env.py`: short docstrings that say *why*, no boilerplate.

## Review Focus

1. **Curriculum scale never reaching the subprocess envs.** If `env_method` is called on the wrong object, training silently stays at scale 0 or 1. The test `test_curriculum_callback_sets_scale_in_every_env` (Task 4) pins this.
2. **Evaluating a normalised model without its `VecNormalize` stats.** The policy then sees un-normalised inputs and every result is garbage. `eval_generalisation` must auto-load `<model stem>_vecnormalize.pkl` and fail loudly if `--normalize` was used in training and the file is missing (`test_load_policy_finds_vecnormalize_next_to_model`, Task 5).
3. **Randomised starts that are physically ungraspable**, for example a corner of the block range combined with arm noise. Training would quietly cap success. `test_scripted_pinch_works_from_randomised_starts` (Task 3) must pass at full L2 range before any training starts.
4. **Zone jitter that changes the observation but not the success check, or the reverse.** `test_success_uses_jittered_zone` (Task 1) pins this.
5. **Block spawning inside the table or the jaws** under randomisation. `test_randomised_reset_rests_block_on_table` (Task 1) pins this.

---

## File structure

| File | Change | Responsibility |
|---|---|---|
| `shelfsort/env.py` | modify | Randomised reset, curriculum scale, per-episode zone positions, 27-D observation |
| `shelfsort/train.py` | modify | CLI flags, vectorised envs, `VecNormalize`, curriculum callback, checkpoints |
| `shelfsort/eval_generalisation.py` | create | Score a model on L0–L3, a wide out-of-distribution level and a radius sweep; write JSON |
| `shelfsort/play.py` | modify | Load `VecNormalize` stats if present (reuses `load_policy` from eval) |
| `tests/test_env.py` | modify | Update the 21-D tests; new randomisation, observation and physics tests |
| `tests/test_train.py` | create | Curriculum callback and train-CLI smoke test |
| `tests/test_eval_generalisation.py` | create | Eval output shape and normaliser loading |
| `results/generalisation/` | create | Committed eval JSONs, final models, training-log excerpts, `README.md` summary |

---

### Task 1: Randomised reset with curriculum scale

**Files:**
- Modify: `shelfsort/env.py` (`__init__`, `reset`, and every `self.SHELF_ZONE_POS[self._active_category]` use at lines ~179, 213, 229, 251)
- Test: `tests/test_env.py`

**Interfaces:**
- Produces:
  - `ShelfSortEnv(scene_path=DEFAULT_SCENE_PATH, reward_mode="sparse", block_xy_range=0.0, arm_qpos_range=0.0, zone_xy_range=0.0)`. Ranges are half-widths of uniform distributions: metres for the block and zones, radians for the first `N_CONTROLLED_JOINTS` (4) arm joints. `wrist_roll` and the gripper are never randomised.
  - `env.set_randomization_scale(scale: float) -> None`. Clips to [0, 1], stores it in `self._randomization_scale`, and multiplies all three ranges at the next reset. Defaults to 1.0, so an env built with ranges is fully randomised unless a curriculum lowers the scale.
  - `env.active_zone_pos -> np.ndarray` (shape (3,)): this episode's (possibly jittered) zone for the active colour. Every reward, success and observation computation uses it instead of `SHELF_ZONE_POS[...]`.
  - `env.reset(seed=None, options={"block_offset": (dx, dy)})`: an explicit block xy offset from `SUPPLY_POS`. It overrides the random draw and is **not** scaled.

- [ ] **Step 1: Write the failing tests** in `tests/test_env.py`

```python
L2 = dict(block_xy_range=0.03, arm_qpos_range=0.1)

def test_zero_ranges_reproduce_fixed_reset():
    env = ShelfSortEnv(block_xy_range=0.0, arm_qpos_range=0.0, zone_xy_range=0.0)
    for seed in range(5):
        env.reset(seed=seed)
        np.testing.assert_allclose(env.data.body(env.BLOCK_BODY_NAMES[env._active_category]).xpos, env.SUPPLY_POS, atol=1e-6)
        np.testing.assert_allclose(env.active_zone_pos, env.SHELF_ZONE_POS[env._active_category])

def test_randomised_block_stays_within_range_and_varies():
    env = ShelfSortEnv(**L2)
    offsets = []
    for seed in range(50):
        env.reset(seed=seed)
        offsets.append(env.data.body(env.BLOCK_BODY_NAMES[env._active_category]).xpos[:2] - env.SUPPLY_POS[:2])
    offsets = np.array(offsets)
    assert np.all(np.abs(offsets) <= 0.03 + 1e-9)
    assert offsets.std(axis=0).min() > 0.01

def test_randomised_arm_pose_within_range_and_ctrl_matches():
    env = ShelfSortEnv(**L2)
    env.reset(seed=3)
    q = np.array([env.data.joint(n).qpos[0] for n in env.ARM_JOINT_NAMES])
    assert np.all(np.abs(q[:4] - env.READY_ARM_QPOS[:4]) <= 0.1 + 1e-9)
    assert q[4] == env.READY_ARM_QPOS[4]
    np.testing.assert_allclose(env._arm_ctrl_target, q)
    np.testing.assert_allclose([env.data.actuator(n).ctrl[0] for n in env.ARM_JOINT_NAMES], q)

def test_same_seed_gives_same_start():
    a, b = ShelfSortEnv(**L2, zone_xy_range=0.02), ShelfSortEnv(**L2, zone_xy_range=0.02)
    oa, _ = a.reset(seed=42)
    ob, _ = b.reset(seed=42)
    np.testing.assert_array_equal(oa, ob)

def test_randomisation_scale_zero_disables_ranges():
    env = ShelfSortEnv(**L2, zone_xy_range=0.02)
    env.set_randomization_scale(0.0)
    env.reset(seed=7)
    np.testing.assert_allclose(env.data.body(env.BLOCK_BODY_NAMES[env._active_category]).xpos, env.SUPPLY_POS, atol=1e-6)
    np.testing.assert_allclose(env.active_zone_pos, env.SHELF_ZONE_POS[env._active_category])

def test_block_offset_option_overrides_random_draw():
    env = ShelfSortEnv(**L2)
    env.reset(seed=0, options={"block_offset": (0.04, -0.02)})
    np.testing.assert_allclose(env.data.body(env.BLOCK_BODY_NAMES[env._active_category]).xpos[:2], env.SUPPLY_POS[:2] + [0.04, -0.02], atol=1e-6)

def test_randomised_reset_rests_block_on_table():
    env = ShelfSortEnv(**L2, zone_xy_range=0.02)
    for seed in range(20):
        env.reset(seed=seed)
        for _ in range(10):
            env.step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        z = env.data.body(env.BLOCK_BODY_NAMES[env._active_category]).xpos[2]
        assert abs(z - env.BLOCK_REST_Z) < 0.002

def test_success_uses_jittered_zone():
    env = ShelfSortEnv(reward_mode="dense", zone_xy_range=0.02)
    env.reset(seed=1)
    assert not np.allclose(env.active_zone_pos, env.SHELF_ZONE_POS[env._active_category])
    fj = env.data.joint(env.BLOCK_FREEJOINT_NAMES[env._active_category])
    fj.qpos[:3] = [*env.active_zone_pos[:2], env.BLOCK_REST_Z]
    fj.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    _, _, terminated, _, info = env.step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    assert info["success"] and terminated
```

- [ ] **Step 2: Run the tests and check they fail**

Run: `uv run pytest tests/test_env.py -k "zero_ranges or randomised or same_seed or scale_zero or block_offset or jittered" -v`
Expected: FAIL (`TypeError: unexpected keyword argument 'block_xy_range'`)

- [ ] **Step 3: Implement it in `shelfsort/env.py`**

In `reset`, after the existing colour draw and fixed placement, draw in this order from `self.np_random`: zone jitter for all three colours (2 values each), then the block offset (2), then arm noise (4). Always draw, even when a range is 0, so the RNG stream doesn't depend on the ranges. Apply `scale` to the ranges, and use `options["block_offset"]` in place of the block draw when it's given. After placing everything, call `mj_forward`, then compute `_prev_reach_dist`, `_prev_transport_dist` and `_prev_height_potential` (move the existing lines below the randomisation). Store the per-episode zones in `self._zone_pos` (a dict). `active_zone_pos` returns `self._zone_pos[self._active_category]`. Set `_arm_ctrl_target` and the arm actuators' `ctrl` to the randomised q.

- [ ] **Step 4: Run the whole suite and check it passes**

Run: `uv run pytest -v`
Expected: all pass. The existing 31 tests still pass, because the defaults are 0.

- [ ] **Step 5: Commit**

```bash
git add shelfsort/env.py tests/test_env.py
git commit -m "Add optional randomised reset with curriculum scale to ShelfSortEnv"
```

---

### Task 2: 27-D observation with relative vectors

**Files:**
- Modify: `shelfsort/env.py` (`__init__` observation space, `_get_obs`)
- Modify: `tests/test_env.py` (`test_observation_space_is_21d_box`, `test_observation_ends_with_active_shelf_position`)

**Interfaces:**
- Consumes: `active_zone_pos` (Task 1).
- Produces: observation layout `[qpos(6), qvel(6), ee_pos(3), block_pos(3), target_pos(3), block_pos − ee_pos(3), target_pos − block_pos(3)]`, float32, shape (27,). `target_pos` is `obs[18:21]`.

- [ ] **Step 1: Update and add tests**

Rename `test_observation_space_is_21d_box` to `test_observation_space_is_27d_box` and assert `(27,)`. Rename `test_observation_ends_with_active_shelf_position` to `test_observation_contains_active_shelf_position`, keep the docstring, and assert `obs[18:21]` against `env.active_zone_pos`. Add:

```python
def test_observation_ends_with_relative_vectors():
    env = ShelfSortEnv(block_xy_range=0.03, arm_qpos_range=0.1, zone_xy_range=0.02)
    obs, _ = env.reset(seed=5)
    ee, block, target = obs[12:15], obs[15:18], obs[18:21]
    np.testing.assert_allclose(obs[21:24], block - ee, atol=1e-6)
    np.testing.assert_allclose(obs[24:27], target - block, atol=1e-6)
```

- [ ] **Step 2: Run them and check they fail**

Run: `uv run pytest tests/test_env.py -k observation -v`
Expected: FAIL (shape is (21,))

- [ ] **Step 3: Implement it.** Change `shape=(21,)` to `(27,)` and append the two differences in `_get_obs`. Add a one-line comment saying that the relative vectors make the grasp and carry look the same wherever the block or zone is.

- [ ] **Step 4: Run the suite and check it passes**

Run: `uv run pytest -v`
Expected: all pass, including `test_env_passes_sb3_check_env`.

- [ ] **Step 5: Commit**

```bash
git add shelfsort/env.py tests/test_env.py
git commit -m "Append block-minus-EE and target-minus-block to the observation (27-D)"
```

---

### Task 3: Physics check — randomised starts are graspable

This runs before any training. The project's first lesson was that physics bugs sank several training runs.

**Files:**
- Modify: `tests/test_env.py`

**Interfaces:**
- Consumes: `ShelfSortEnv(**L2)`, `reset(options={"block_offset": ...})` (Task 1).

- [ ] **Step 1: Write the test**

Pull the `goto`/re-aim/descend/close sequence from `test_scripted_pinch_grasps_and_lifts_the_block` into a module-level helper `_scripted_pinch(env) -> dict` that returns the last `info`, and make the existing test call it (same assertions). Then add:

```python
CORNERS = [(dx, dy) for dx in (-0.03, 0.03) for dy in (-0.03, 0.03)]

@pytest.mark.parametrize("seed", range(12))
def test_scripted_pinch_works_from_randomised_starts(seed):
    env = ShelfSortEnv(block_xy_range=0.03, arm_qpos_range=0.1)
    options = {"block_offset": CORNERS[seed]} if seed < len(CORNERS) else None
    env.reset(seed=seed, options=options)
    info = _scripted_pinch(env)
    assert info["grasped"]
    assert env.data.body(env.BLOCK_BODY_NAMES[env._active_category]).xpos[2] > 0.04
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_env.py -k "scripted_pinch" -v`
Expected: PASS for all 12 cases plus the original test.

**STOP condition:** if any corner case fails, do **not** shrink the range or change the helper to make it pass. Find out whether the cause is IK reach or a jaw-alignment problem (print the final EE–block offset and the gripper angle), commit the failing test marked `xfail` with the reason, and record it in `results/generalisation/README.md`. If more than 2 of the 12 fail, stop the plan and report back. Training on unreachable starts would hide the real result.

- [ ] **Step 3: Commit**

```bash
git add tests/test_env.py
git commit -m "Check the scripted pinch grasps from randomised block and arm starts"
```

---

### Task 4: Training script — vec envs, VecNormalize, curriculum, checkpoints

**Files:**
- Modify: `shelfsort/train.py`
- Create: `tests/test_train.py`

**Interfaces:**
- Consumes: `ShelfSortEnv(..., block_xy_range, arm_qpos_range, zone_xy_range)`, `set_randomization_scale` (Task 1).
- Produces:
  - `class RandomizationCurriculum(BaseCallback)` with `__init__(self, ramp_steps: int)`. In `_on_rollout_start` it sets `scale = min(1.0, self.num_timesteps / ramp_steps)` in every env via `self.training_env.env_method("set_randomization_scale", scale)` and logs `curriculum/scale`. It also sets the scale once in `_on_training_start` (scale 0 at step 0).
  - `make_env(reward_mode: str, block_xy_range: float, arm_qpos_range: float, zone_xy_range: float) -> Callable[[], gym.Env]`. It returns a thunk that builds a `Monitor`-wrapped `ShelfSortEnv`.
  - New CLI flags (existing flags unchanged): `--n-envs` (int, default 1), `--block-xy-range` / `--arm-qpos-range` / `--zone-xy-range` (float, default 0.0), `--curriculum-steps` (int, default 0 = no curriculum, so full ranges from the start), `--normalize` (flag: wrap in `VecNormalize(norm_obs=True, norm_reward=True)`), `--checkpoint-every` (int, default 0 = off), `--run-name` (str, default `f"{reward_mode}_seed{seed}"`).
  - Outputs: `<out>/models/<run_name>.zip`, `<out>/models/<run_name>_vecnormalize.pkl` (only with `--normalize`), `<out>/checkpoints/<run_name>/` (`CheckpointCallback`, with `save_freq = max(1, checkpoint_every // n_envs)` and `save_vecnormalize=True`), `<out>/tb/`.
  - Build the env with `SubprocVecEnv` when `n_envs > 1`, otherwise `DummyVecEnv`. Seed it with `seed`. Use `PPO(..., device="cpu")`.

- [ ] **Step 1: Write the failing tests** in `tests/test_train.py`

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from shelfsort.train import RandomizationCurriculum, make_env


def test_curriculum_callback_sets_scale_in_every_env():
    venv = DummyVecEnv([make_env("dense", 0.03, 0.1, 0.0) for _ in range(2)])
    model = PPO("MlpPolicy", venv, n_steps=32, batch_size=32, n_epochs=1, device="cpu")
    model.learn(total_timesteps=128, callback=RandomizationCurriculum(ramp_steps=256))
    scales = venv.get_attr("_randomization_scale")  # get_attr reaches through Monitor
    assert all(0.0 < s < 1.0 for s in scales)
    assert len(set(scales)) == 1


def test_train_cli_writes_model_and_vecnormalize(tmp_path):
    import subprocess, sys
    subprocess.run(
        [sys.executable, "-m", "shelfsort.train", "--reward-mode", "dense", "--timesteps", "256",
         "--n-envs", "2", "--n-steps", "64", "--block-xy-range", "0.03", "--curriculum-steps", "128", "--normalize",
         "--out", str(tmp_path), "--run-name", "smoke"],
        check=True,
    )
    assert (tmp_path / "models" / "smoke.zip").exists()
    assert (tmp_path / "models" / "smoke_vecnormalize.pkl").exists()
```

Task 1 stores the scale as `self._randomization_scale`. Add an `--n-steps` flag (int, default 2048 = SB3 default) so the smoke test can use small rollouts. Training runs in Task 6 never pass it.

- [ ] **Step 2: Run them and check they fail**

Run: `uv run pytest tests/test_train.py -v`
Expected: FAIL (`ImportError: cannot import name 'RandomizationCurriculum'`)

- [ ] **Step 3: Implement it in `shelfsort/train.py`.** `SuccessLogger` stays as it is (it already loops over `infos`). Save the `VecNormalize` stats with `env.save(path)` after `model.save`. Print the measured fps at the end (`timesteps / wall seconds`).

- [ ] **Step 4: Run the full suite and check it passes**

Run: `uv run pytest -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add shelfsort/train.py tests/test_train.py
git commit -m "Add vectorised envs, VecNormalize, randomisation curriculum and checkpoints to train.py"
```

---

### Task 5: Generalisation evaluation script

**Files:**
- Create: `shelfsort/eval_generalisation.py`
- Modify: `shelfsort/play.py` (use `load_policy` so normalised models play correctly)
- Create: `tests/test_eval_generalisation.py`

**Interfaces:**
- Consumes: `ShelfSortEnv` kwargs and the `block_offset` option (Task 1), and the files written by Task 4.
- Produces:
  - `LEVELS: dict[str, dict]`, exactly:
    - `"L0": {}`
    - `"L1": {"block_xy_range": 0.03}`
    - `"L2": {"block_xy_range": 0.03, "arm_qpos_range": 0.1}`
    - `"L3": {"block_xy_range": 0.03, "arm_qpos_range": 0.1, "zone_xy_range": 0.02}`
    - `"OOD_wide": {"block_xy_range": 0.05, "arm_qpos_range": 0.15, "zone_xy_range": 0.03}`
  - `RADII = (0.01, 0.02, 0.03, 0.04, 0.05)`. For each radius, every episode puts the block at that distance in a random direction, using `options={"block_offset": ...}`. The angle comes from `np.random.default_rng(seed)`. This sweep is deterministic only and has no arm or zone noise.
  - `load_policy(model_path: Path) -> tuple[PPO, VecNormalize | None]`. If `<model_path.stem>_vecnormalize.pkl` exists next to the model, load it with `training=False, norm_reward=False`. If the model's observation space is 27-D and no stats file exists, emit a `warnings.warn` (models trained without `--normalize` are valid).
  - `evaluate(model_path: Path, env_kwargs: dict, deterministic: bool, seeds: Iterable[int], block_radius: float | None = None) -> dict`. It returns `{"mean": float, "per_colour": {c: float}, "grasp": float, "mean_steps_to_place": {c: float | None}, "episodes_per_colour": {c: int}}`. `mean` is the mean of the per-colour rates.
  - CLI: `uv run python -m shelfsort.eval_generalisation --model PATH --out JSON [--episodes 300] [--workers 4]`. It runs every level both deterministic and stochastic, plus the radius sweep, over seeds `5000 .. 5000+episodes−1`, using a `ProcessPoolExecutor`. It writes a JSON list of `{"name", "mode", **evaluate_result}` and prints one line per job: `L1  det   mean=0.15  R/G/B=0.27/0.11/0.08  grasp=0.26`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from shelfsort.env import ShelfSortEnv
from shelfsort.eval_generalisation import LEVELS, evaluate, load_policy


def _tiny_model(tmp_path, normalize):
    venv = DummyVecEnv([lambda: ShelfSortEnv(reward_mode="dense")])
    if normalize:
        venv = VecNormalize(venv)
    model = PPO("MlpPolicy", venv, n_steps=32, batch_size=32, n_epochs=1, device="cpu")
    model.learn(64)
    model.save(tmp_path / "m")
    if normalize:
        venv.save(tmp_path / "m_vecnormalize.pkl")
    return tmp_path / "m.zip"


def test_load_policy_finds_vecnormalize_next_to_model(tmp_path):
    _, norm = load_policy(_tiny_model(tmp_path, normalize=True))
    assert norm is not None and not norm.training


def test_evaluate_returns_per_colour_rates(tmp_path):
    path = _tiny_model(tmp_path, normalize=False)
    out = evaluate(path, LEVELS["L2"], deterministic=True, seeds=range(5000, 5006))
    assert set(out["per_colour"]) == set(ShelfSortEnv.BLOCK_CATEGORIES)
    assert 0.0 <= out["mean"] <= 1.0 and 0.0 <= out["grasp"] <= 1.0
    assert sum(out["episodes_per_colour"].values()) == 6


def test_evaluate_radius_sweep_places_block_at_radius(tmp_path, monkeypatch):
    path = _tiny_model(tmp_path, normalize=False)
    seen = []
    orig = ShelfSortEnv.reset
    def spy(self, seed=None, options=None):
        seen.append(np.linalg.norm(options["block_offset"]))
        return orig(self, seed=seed, options=options)
    monkeypatch.setattr(ShelfSortEnv, "reset", spy)
    evaluate(path, {}, deterministic=True, seeds=range(3), block_radius=0.02)
    np.testing.assert_allclose(seen, 0.02)
```

- [ ] **Step 2: Run them and check they fail**

Run: `uv run pytest tests/test_eval_generalisation.py -v`
Expected: FAIL (`ModuleNotFoundError: shelfsort.eval_generalisation`)

- [ ] **Step 3: Implement `shelfsort/eval_generalisation.py`.** Before each `model.predict`, apply `norm.normalize_obs(obs)` when a normaliser is present. `steps to place` is the step count of successful episodes only (`None` if a colour never succeeds). Also change `play.py` to use `load_policy` and normalise observations in the same way.

- [ ] **Step 4: Run the full suite and check it passes**

Run: `uv run pytest -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add shelfsort/eval_generalisation.py shelfsort/play.py tests/test_eval_generalisation.py
git commit -m "Add generalisation evaluation over randomised and out-of-distribution starts"
```

---

### Task 6: Run the experiments and commit results

These are long-running jobs. Run each training command in the background (`run_in_background`, or `nohup … > log 2>&1 &`) and wait for it to finish instead of polling every few seconds. Check back about every 10–20 minutes. `runs/` is gitignored, so everything worth keeping gets copied into `results/generalisation/`.

**Files:**
- Create: `results/generalisation/README.md`, `results/generalisation/<run_name>/{eval.json, model.zip, vecnormalize.pkl, train_tail.log}`

- [ ] **Step 1: Environment and throughput check (E0)**

```bash
nproc
uv run python -m shelfsort.train --reward-mode dense --timesteps 40000 --n-envs 4 \
  --block-xy-range 0.03 --arm-qpos-range 0.1 --curriculum-steps 20000 --normalize \
  --out runs/gen_smoke --run-name smoke
```
Expected: it finishes and prints its fps. Write down `nproc` and the fps, then work out the wall time of one 5M run (`5e6 / fps`). Reference: about 1,260 steps/s on a 4-core i5 Mac, which is about 66 min per 5M.

If `nproc ≥ 12`, you may run the 3 seeds of Step 3 at the same time (each one uses 4 envs). Otherwise run them one after another.

- [ ] **Step 2: Pilot run (E1): dense, L2, curriculum, seed 0, 5M**

```bash
uv run python -m shelfsort.train --reward-mode dense --timesteps 5000000 --n-envs 4 \
  --block-xy-range 0.03 --arm-qpos-range 0.1 --curriculum-steps 2000000 --normalize \
  --checkpoint-every 500000 --seed 0 --out runs/gen --run-name dense_L2_seed0 > runs/gen_dense_L2_seed0.log 2>&1
uv run python -m shelfsort.eval_generalisation --model runs/gen/models/dense_L2_seed0.zip \
  --out results/generalisation/dense_L2_seed0/eval.json
```

Then copy in `model.zip`, `vecnormalize.pkl` and the last 200 lines of the log (`train_tail.log`), and commit them.

**Decision gate after E1.** Use the L2 deterministic mean from `eval.json`:
- **≥ 0.6:** the approach works. Go to Step 3.
- **0.3–0.6, or `rollout/success_rate` still rising over the last 1M steps:** rerun E1 with `--timesteps 10000000 --curriculum-steps 4000000` as `dense_L2_seed0_10M`, evaluate it, then continue to Step 3 using 10M for the seeds.
- **< 0.3 and flat:** do not run seeds. Look at the TensorBoard scalars (`rollout/grasp_*`, `curriculum/scale`, `train/approx_kl`). Write up the findings in the README (Step 5), push, and stop. Don't tune hyperparameters or the reward on your own.

- [ ] **Step 3: Seeds (E2): repeat E1 with `--seed 1` and `--seed 2`** (and the run names `dense_L2_seed1`, `dense_L2_seed2`). Use the same timesteps the gate settled on. Evaluate and commit each one as in Step 2.

- [ ] **Step 4: Zone jitter (E3): one run on L3, seed 0.** This is the same command as E1 plus `--zone-xy-range 0.02`, with the run name `dense_L3_seed0`. Evaluate and commit it. Skip this step if E1 failed the gate.

- [ ] **Step 5: Write `results/generalisation/README.md`**

Include:
1. Machine: `nproc` and the measured fps.
2. The baseline table from *Background*, copied unchanged and labelled "v6 3M, trained on fixed starts".
3. One table per run with the same rows (L0, L1, L2, L3, OOD_wide, det and stoch; the radius sweep det), so it can be compared with the baseline directly.
4. For E2: mean ± std over the 3 seeds for each L-level, deterministic and stochastic.
5. What got better and what didn't. Note whether the fixed-start (L0) performance held up, and which colour is still weakest.
6. Anything from the Task 3 STOP condition.

Report numbers only from `eval.json` files. Do not estimate.

- [ ] **Step 6: Commit and push**

```bash
git add results/generalisation
git commit -m "Add randomised-reset generalisation results"
git push fork feat/randomised-reset
```

---

## Time budget (4 envs at about 1,260 steps/s; rescale using the E0 fps)

| Step | Approx. wall time |
|---|---|
| Tasks 1–5 (code + tests) | 30–60 min |
| E0 smoke | 1 min |
| E1 pilot 5M + eval | ~70 min |
| E2 two more seeds | ~2 h 20 min (sequential) or ~70 min (parallel, if `nproc ≥ 12`) |
| E3 L3 run | ~70 min |
| **Total** | **about 4.5–5.5 h** (doubles at the 10M gate branch) |
