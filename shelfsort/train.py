"""Train PPO on ShelfSortEnv. Usage: uv run python -m shelfsort.train --reward-mode dense --timesteps 50000"""
import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from shelfsort.env import ShelfSortEnv


class SuccessLogger(BaseCallback):
    """Logs per-episode success and grasp flags (from `info`) to TensorBoard."""

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self.logger.record_mean("rollout/success_rate", float(info["success"]))
                self.logger.record_mean("rollout/grasp_rate", float(info["ever_grasped"]))
        return True


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
