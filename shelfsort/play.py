"""Watch a trained policy in the MuJoCo viewer.

Usage (macOS needs mjpython for the viewer):
    uv run mjpython -m shelfsort.play --model runs/models/dense_seed0.zip
    uv run python -m shelfsort.play --no-viewer --episodes 20   # headless success rate
"""
import argparse
import time
from pathlib import Path

import mujoco.viewer
from stable_baselines3 import PPO

from shelfsort.env import ShelfSortEnv


def run_episode(env, model, deterministic, viewer=None):
    obs, _ = env.reset()
    done, total, info = False, 0.0, {}
    while not done and (viewer is None or viewer.is_running()):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        done = terminated or truncated
        if viewer is not None:
            viewer.sync()
            time.sleep(1 / 60)
    return total, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("runs/models/dense_seed0.zip"))
    parser.add_argument("--reward-mode", choices=ShelfSortEnv.REWARD_MODES, default="dense")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of using the mean")
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    env = ShelfSortEnv(reward_mode=args.reward_mode)
    model = PPO.load(args.model)
    deterministic = not args.stochastic

    successes = 0
    if args.no_viewer:
        for i in range(args.episodes):
            total, info = run_episode(env, model, deterministic)
            successes += int(info["success"])
            print(f"episode {i}: return={total:.2f} success={info['success']} grasped={info['ever_grasped']}")
    else:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            for i in range(args.episodes):
                if not viewer.is_running():
                    break
                total, info = run_episode(env, model, deterministic, viewer)
                successes += int(info["success"])
                print(f"episode {i}: return={total:.2f} success={info['success']} grasped={info['ever_grasped']}")
    print(f"success rate: {successes}/{args.episodes}")


if __name__ == "__main__":
    main()
