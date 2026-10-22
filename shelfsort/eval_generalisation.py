"""Score a trained policy on fixed, randomised and out-of-distribution starts.

Usage: uv run python -m shelfsort.eval_generalisation --model runs/gen/models/dense_L2_seed0.zip --out eval.json
"""
import argparse
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from shelfsort.env import ShelfSortEnv

LEVELS = {
    "L0": {},
    "L1": {"block_xy_range": 0.03},
    "L2": {"block_xy_range": 0.03, "arm_qpos_range": 0.1},
    "L3": {"block_xy_range": 0.03, "arm_qpos_range": 0.1, "zone_xy_range": 0.02},
    "OOD_wide": {"block_xy_range": 0.05, "arm_qpos_range": 0.15, "zone_xy_range": 0.03},
}
RADII = (0.01, 0.02, 0.03, 0.04, 0.05)
FIRST_SEED = 5000


def load_policy(model_path):
    """Loads the model and, if saved next to it, its VecNormalize stats frozen for evaluation.

    A normalised policy fed raw observations still runs but scores garbage, so a 27-D model
    without stats gets a warning (models trained without --normalize are valid).
    """
    model_path = Path(model_path)
    model = PPO.load(model_path, device="cpu")
    stats_path = model_path.with_name(f"{model_path.stem}_vecnormalize.pkl")
    if not stats_path.exists():  # results/generalisation/<run>/ stores model.zip beside vecnormalize.pkl
        stats_path = model_path.with_name("vecnormalize.pkl")
    if not stats_path.exists():
        if model.observation_space.shape == (27,):
            warnings.warn(f"no VecNormalize stats at {stats_path}; evaluating on raw observations")
        return model, None
    norm = VecNormalize.load(str(stats_path), DummyVecEnv([ShelfSortEnv]))
    norm.training = False
    norm.norm_reward = False
    return model, norm


def evaluate(model_path, env_kwargs, deterministic, seeds, block_radius=None):
    """Runs one episode per seed. With `block_radius`, the block sits that far from SUPPLY_POS in a
    seed-dependent direction."""
    model, norm = load_policy(model_path)
    env = ShelfSortEnv(reward_mode="dense", **env_kwargs)
    success = {c: [] for c in ShelfSortEnv.BLOCK_CATEGORIES}
    grasp = {c: [] for c in ShelfSortEnv.BLOCK_CATEGORIES}
    steps = {c: [] for c in ShelfSortEnv.BLOCK_CATEGORIES}

    for seed in seeds:
        options = None
        if block_radius is not None:
            angle = np.random.default_rng(seed).uniform(0.0, 2 * np.pi)
            options = {"block_offset": (block_radius * np.cos(angle), block_radius * np.sin(angle))}
        obs, _ = env.reset(seed=seed, options=options)
        done, n, info = False, 0, {}
        while not done:
            policy_obs = norm.normalize_obs(obs) if norm is not None else obs
            action, _ = model.predict(policy_obs, deterministic=deterministic)
            obs, _, terminated, truncated, info = env.step(action)
            n += 1
            done = terminated or truncated
        category = info["category"]
        success[category].append(float(info["success"]))
        grasp[category].append(float(info["ever_grasped"]))
        if info["success"]:
            steps[category].append(n)

    per_colour = {c: float(np.mean(v)) if v else 0.0 for c, v in success.items()}
    all_grasps = [g for v in grasp.values() for g in v]
    return {
        "mean": float(np.mean(list(per_colour.values()))),
        "per_colour": per_colour,
        "grasp": float(np.mean(all_grasps)) if all_grasps else 0.0,
        "mean_steps_to_place": {c: float(np.mean(v)) if v else None for c, v in steps.items()},
        "episodes_per_colour": {c: len(v) for c, v in success.items()},
    }


def _run_job(job):
    torch.set_num_threads(1)  # the workers already use every core
    name, mode, model_path, env_kwargs, block_radius, seeds = job
    result = evaluate(model_path, env_kwargs, mode == "det", seeds, block_radius)
    return {"name": name, "mode": mode, **result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    seeds = list(range(FIRST_SEED, FIRST_SEED + args.episodes))
    jobs = [(name, mode, args.model, kwargs, None, seeds) for name, kwargs in LEVELS.items() for mode in ("det", "stoch")]
    jobs += [(f"radius_{r:.2f}", "det", args.model, {}, r, seeds) for r in RADII]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_run_job, jobs))

    for r in results:
        rgb = "/".join(f"{r['per_colour'][c]:.2f}" for c in ShelfSortEnv.BLOCK_CATEGORIES)
        print(f"{r['name']:<11} {r['mode']:<5} mean={r['mean']:.2f}  R/G/B={rgb}  grasp={r['grasp']:.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
