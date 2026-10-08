"""Collect the SAC numbers for the Part-C report (Tables 3-4, Figure 1) from a training output folder.

Usage: uv run --with matplotlib python -m shelfsort.report --runs runs/main_sac --out docs/report_sac
"""
import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from shelfsort.env import ShelfSortEnv
from shelfsort.evaluate import evaluate

COLOURS = ShelfSortEnv.BLOCK_CATEGORIES


def scalars(tb_dir):
    acc = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    acc.Reload()
    return {tag: np.array([(s.step, s.value) for s in acc.Scalars(tag)]) for tag in acc.Tags()["scalars"]}


def first_step(series, threshold):
    hits = series[series[:, 1] >= threshold] if series is not None else []
    return int(hits[0, 0]) if len(hits) else None


def training_table(tb):
    """Table 3 metrics. The rollout rates are SAC's stochastic training episodes, last 50 per colour."""
    per_colour = {c: tb.get(f"rollout/success_{c}") for c in COLOURS}
    firsts = [first_step(s, 1e-9) for s in per_colour.values() if s is not None]
    return {
        "total_steps": int(tb["rollout/ep_len_mean"][-1, 0]),
        "success_mean": float(tb["rollout/success_rate"][-1, 1]) if "rollout/success_rate" in tb else 0.0,
        "success_per_colour": {c: float(s[-1, 1]) if s is not None else 0.0 for c, s in per_colour.items()},
        "grasp_final": float(tb["rollout/grasp_rate"][-1, 1]) if "rollout/grasp_rate" in tb else 0.0,
        "grasp_peak": float(tb["rollout/grasp_rate"][:, 1].max()) if "rollout/grasp_rate" in tb else 0.0,
        "first_success_step": min([f for f in firsts if f is not None], default=None),
        "steps_to_50pct": {c: first_step(s, 0.5) if s is not None else None for c, s in per_colour.items()},
        "mean_episode_length": float(tb["rollout/ep_len_mean"][-1, 1]),
    }


def checkpoint_scan(runs, run_name, episodes):
    """Deterministic success of every saved checkpoint, to show how stable training was."""
    paths = sorted((runs / "checkpoints" / run_name).glob("*.zip"), key=lambda p: int(p.stem.split("_")[-2]))
    scan = {}
    for path in paths:
        r = evaluate(SAC.load(path), "dense", episodes, deterministic=True, seed_base=90_000)
        scan[int(path.stem.split("_")[-2])] = r["mean_of_colours"]
        print(f"checkpoint {path.name}: {r['mean_of_colours']:.2f}", flush=True)
    return scan


def plot(curves, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for mode, tb in curves.items():
        if "rollout/success_rate" in tb:
            s = tb["rollout/success_rate"]
            axes[0].plot(s[:, 0] / 1e6, s[:, 1], label=f"{mode} SAC", linestyle="-" if mode == "dense" else "--")
        if "rollout/grasp_rate" in tb:
            g = tb["rollout/grasp_rate"]
            axes[1].plot(g[:, 0] / 1e6, g[:, 1], label=f"{mode} SAC grasp rate",
                         linestyle="-" if mode == "dense" else "--")
    dense = curves.get("dense", {})
    for colour in COLOURS:
        if f"rollout/success_{colour}" in dense:
            s = dense[f"rollout/success_{colour}"]
            axes[1].plot(s[:, 0] / 1e6, s[:, 1], label=f"dense success {colour}", color=colour, alpha=0.7)
    axes[0].set_title("Success rate (mean of colours, last 50 episodes each)")
    axes[1].set_title("Dense SAC per-colour success, and grasp rate")
    for ax in axes:
        ax.set_xlabel("environment steps (millions)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("rate")
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs/main_sac"))
    parser.add_argument("--out", type=Path, default=Path("docs/report_sac"))
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--scan-episodes", type=int, default=60)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    results, curves = {}, {}
    for mode in ShelfSortEnv.REWARD_MODES:
        run_name = f"{mode}_seed0_sac"
        tb_dirs = sorted((args.runs / "tb").glob(f"{run_name}_*"), key=lambda d: d.stat().st_mtime)
        model_path = args.runs / "models" / f"{run_name}.zip"
        if not tb_dirs or not model_path.exists():
            print(f"skipping {mode}: run not finished")
            continue
        curves[mode] = tb = scalars(tb_dirs[-1])
        model = SAC.load(model_path)
        results[mode] = {
            "training": training_table(tb),
            "final_policy": {m: evaluate(model, mode, args.episodes, m == "deterministic")
                             for m in ("deterministic", "stochastic")},
        }
        if mode == "dense":
            results[mode]["checkpoint_scan"] = checkpoint_scan(args.runs, run_name, args.scan_episodes)
        print(mode, json.dumps(results[mode]["final_policy"]), flush=True)

    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    plot(curves, args.out / "training_curves.png")


if __name__ == "__main__":
    main()
