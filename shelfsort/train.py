"""Train PPO or SAC on ShelfSortEnv. Usage: uv run python -m shelfsort.train --reward-mode dense --algo sac --timesteps 50000"""
import argparse
import os
from collections import deque
from pathlib import Path

# Each SubprocVecEnv worker would otherwise start a full pool of BLAS/torch threads, which
# oversubscribes the CPU badly enough to freeze the machine.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from shelfsort.env import ShelfSortEnv

ALGOS = {"ppo": PPO, "sac": SAC}


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
    parser.add_argument("--algo", choices=ALGOS, default="ppo")
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--n-envs", type=int, default=1, help="parallel environments (SAC trains faster with 4)")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="env steps between saved checkpoints, 0 = off")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    args = parser.parse_args()
    torch.set_num_threads(2)

    # PPO keeps the original run name so existing checkpoints and TensorBoard logs still line up.
    run_name = f"{args.reward_mode}_seed{args.seed}" + ("_sac" if args.algo == "sac" else "")
    if args.n_envs > 1:
        env = make_vec_env(ShelfSortEnv, n_envs=args.n_envs, seed=args.seed, vec_env_cls=SubprocVecEnv,
                           env_kwargs={"reward_mode": args.reward_mode})
    else:
        env = Monitor(ShelfSortEnv(reward_mode=args.reward_mode))
    model = ALGOS[args.algo]("MlpPolicy", env, seed=args.seed, verbose=1, tensorboard_log=str(args.out / "tb"))

    callbacks = [SuccessLogger()]
    if args.checkpoint_every:
        callbacks.append(CheckpointCallback(save_freq=max(args.checkpoint_every // args.n_envs, 1),
                                            save_path=str(args.out / "checkpoints" / run_name), name_prefix=run_name))
    model.learn(total_timesteps=args.timesteps, callback=CallbackList(callbacks), tb_log_name=run_name)
    model.save(args.out / "models" / run_name)


if __name__ == "__main__":
    main()
