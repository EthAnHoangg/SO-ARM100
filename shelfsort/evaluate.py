"""Headless per-colour evaluation of a trained policy, deterministic and stochastic.

Usage: uv run python -m shelfsort.evaluate --model runs/main_sac/models/dense_seed0_sac.zip --episodes 300
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC

from shelfsort.env import ShelfSortEnv

EVAL_SEED_BASE = 100_000  # fixed so every policy is tested on the same episodes


def evaluate(model, reward_mode, episodes, deterministic, seed_base=EVAL_SEED_BASE):
    env = ShelfSortEnv(reward_mode=reward_mode)
    success, steps, grasped = defaultdict(list), defaultdict(list), []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed_base + i)
        colour = env._active_category
        for t in range(1, env.MAX_EPISODE_STEPS + 1):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        success[colour].append(info["success"])
        grasped.append(info["ever_grasped"])
        if info["success"]:
            steps[colour].append(t)
    per_colour = {c: float(np.mean(success[c])) for c in ShelfSortEnv.BLOCK_CATEGORIES}
    return {
        "mean_of_colours": float(np.mean(list(per_colour.values()))),
        "per_colour": per_colour,
        "episodes_per_colour": {c: len(success[c]) for c in ShelfSortEnv.BLOCK_CATEGORIES},
        "mean_steps_to_place": {c: float(np.mean(steps[c])) if steps[c] else None
                                for c in ShelfSortEnv.BLOCK_CATEGORIES},
        "grasp_rate": float(np.mean(grasped)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reward-mode", choices=ShelfSortEnv.REWARD_MODES, default="dense")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--json", type=Path, default=None, help="also write the results to this file")
    args = parser.parse_args()

    model = (SAC if "_sac" in args.model.name else PPO).load(args.model)
    results = {}
    for mode in ("deterministic", "stochastic"):
        r = results[mode] = evaluate(model, args.reward_mode, args.episodes, mode == "deterministic")
        colours = " / ".join(f"{c} {r['per_colour'][c]:.2f}" for c in ShelfSortEnv.BLOCK_CATEGORIES)
        print(f"{mode:13s} mean {r['mean_of_colours']:.2f} | {colours} | grasp {r['grasp_rate']:.2f}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
