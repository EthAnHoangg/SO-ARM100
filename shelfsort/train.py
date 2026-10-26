"""Train PPO or SAC on ShelfSortEnv. Usage: uv run python -m shelfsort.train --reward-mode dense --algo sac --timesteps 50000"""
import argparse
import os
import time
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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

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


class RandomizationCurriculum(BaseCallback):
    """Ramps the start-state randomisation linearly from 0 to full over `ramp_steps`.

    The scale is pushed through `env_method` so it reaches envs living in subprocesses; setting it
    on a local env object would leave training silently at the initial scale. SAC starts a rollout
    every step, so the scale is only re-sent once it has moved by `MIN_DELTA`.
    """

    MIN_DELTA = 1e-3

    def __init__(self, ramp_steps):
        super().__init__()
        self.ramp_steps = ramp_steps
        self._sent = None

    def _set_scale(self):
        scale = min(1.0, self.num_timesteps / self.ramp_steps)
        if self._sent is not None and scale - self._sent < self.MIN_DELTA and not (scale == 1.0 > self._sent):
            return
        self.training_env.env_method("set_randomization_scale", scale)
        self.logger.record("curriculum/scale", scale)
        self._sent = scale

    def _on_training_start(self):
        self._set_scale()

    def _on_rollout_start(self):
        self._set_scale()

    def _on_step(self):
        return True


def make_env(reward_mode, block_xy_range, arm_qpos_range, zone_xy_range):
    def thunk():
        return Monitor(
            ShelfSortEnv(
                reward_mode=reward_mode,
                block_xy_range=block_xy_range,
                arm_qpos_range=arm_qpos_range,
                zone_xy_range=zone_xy_range,
            )
        )

    return thunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-mode", choices=ShelfSortEnv.REWARD_MODES, default="sparse")
    parser.add_argument("--algo", choices=ALGOS, default="ppo")
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--n-envs", type=int, default=1, help="parallel environments (SAC trains faster with 4)")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="env steps between saved checkpoints, 0 = off")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--n-steps", type=int, default=2048, help="PPO rollout length per env")
    parser.add_argument("--block-xy-range", type=float, default=0.0)
    parser.add_argument("--arm-qpos-range", type=float, default=0.0)
    parser.add_argument("--zone-xy-range", type=float, default=0.0)
    parser.add_argument("--curriculum-steps", type=int, default=0, help="0 = full ranges from the start")
    parser.add_argument("--normalize", action="store_true", help="wrap in VecNormalize (obs and reward)")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    torch.set_num_threads(2)

    # PPO keeps the original run name so existing checkpoints and TensorBoard logs still line up.
    run_name = args.run_name or f"{args.reward_mode}_seed{args.seed}" + ("_sac" if args.algo == "sac" else "")
    env_fns = [
        make_env(args.reward_mode, args.block_xy_range, args.arm_qpos_range, args.zone_xy_range)
        for _ in range(args.n_envs)
    ]
    env = SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns)
    env.seed(args.seed)
    if args.normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

    callbacks = [SuccessLogger()]
    if args.curriculum_steps > 0:
        callbacks.append(RandomizationCurriculum(args.curriculum_steps))
    if args.checkpoint_every > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(1, args.checkpoint_every // args.n_envs),
                save_path=str(args.out / "checkpoints" / run_name),
                name_prefix=run_name,
                save_vecnormalize=True,
            )
        )

    algo_kwargs = {"n_steps": args.n_steps} if args.algo == "ppo" else {}
    model = ALGOS[args.algo](
        "MlpPolicy",
        env,
        seed=args.seed,
        verbose=1,
        tensorboard_log=str(args.out / "tb"),
        device="cpu",
        **algo_kwargs,
    )
    start = time.perf_counter()
    model.learn(total_timesteps=args.timesteps, callback=CallbackList(callbacks), tb_log_name=run_name)
    elapsed = time.perf_counter() - start
    model.save(args.out / "models" / run_name)
    if args.normalize:
        env.save(args.out / "models" / f"{run_name}_vecnormalize.pkl")
    env.close()
    print(f"fps: {model.num_timesteps / elapsed:.0f} ({model.num_timesteps} timesteps in {elapsed:.0f} s)")


if __name__ == "__main__":
    main()
