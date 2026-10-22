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
