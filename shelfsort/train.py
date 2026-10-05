"""Train PPO on ShelfSortEnv. Usage: uv run python -m shelfsort.train --reward-mode dense --timesteps 50000"""
import argparse
from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from shelfsort.env import ShelfSortEnv


class SuccessLogger(BaseCallback):
    """Logs per-episode success and grasp flags (from `info`) to TensorBoard.

    `success_rate` is the mean of the per-category rates, not the raw episode mean, so a policy
    that only solves one of three categories reads as ~33% regardless of how often it is drawn.
    """

    WINDOW = 50  # most recent episodes kept per category

    def _init_callback(self):
        self._success = {c: deque(maxlen=self.WINDOW) for c in ShelfSortEnv.BLOCK_CATEGORIES}
        self._grasp = {c: deque(maxlen=self.WINDOW) for c in ShelfSortEnv.BLOCK_CATEGORIES}

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self._success[info["category"]].append(float(info["success"]))
                self._grasp[info["category"]].append(float(info["ever_grasped"]))
        return True

    def _on_rollout_end(self):
        success = {c: np.mean(v) for c, v in self._success.items() if v}
        grasp = {c: np.mean(v) for c, v in self._grasp.items() if v}
        for c in success:
            self.logger.record(f"rollout/success_{c}", success[c])
            self.logger.record(f"rollout/grasp_{c}", grasp[c])
        if len(success) == len(ShelfSortEnv.BLOCK_CATEGORIES):
            self.logger.record("rollout/success_rate", float(np.mean(list(success.values()))))
            self.logger.record("rollout/grasp_rate", float(np.mean(list(grasp.values()))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-mode", choices=ShelfSortEnv.REWARD_MODES, default="sparse")
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    args = parser.parse_args()

    run_name = f"{args.reward_mode}_seed{args.seed}"
    env = Monitor(ShelfSortEnv(reward_mode=args.reward_mode))
    model = PPO("MlpPolicy", env, seed=args.seed, verbose=1, tensorboard_log=str(args.out / "tb"))
    model.learn(total_timesteps=args.timesteps, callback=SuccessLogger(), tb_log_name=run_name)
    model.save(args.out / "models" / run_name)


if __name__ == "__main__":
    main()
