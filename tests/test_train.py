from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from shelfsort.train import RandomizationCurriculum, make_env


def test_curriculum_callback_sets_scale_in_every_env():
    venv = DummyVecEnv([make_env("dense", 0.03, 0.1, 0.0) for _ in range(2)])
    model = PPO("MlpPolicy", venv, n_steps=32, batch_size=32, n_epochs=1, device="cpu")
    model.learn(total_timesteps=128, callback=RandomizationCurriculum(ramp_steps=256))
    scales = venv.get_attr("_randomization_scale")  # get_attr reaches through Monitor
    assert all(0.0 < s < 1.0 for s in scales)
    assert len(set(scales)) == 1


def test_curriculum_callback_reaches_full_scale_under_sac():
    venv = DummyVecEnv([make_env("dense", 0.03, 0.1, 0.0) for _ in range(2)])
    model = SAC("MlpPolicy", venv, learning_starts=16, device="cpu")
    model.learn(total_timesteps=96, callback=RandomizationCurriculum(ramp_steps=64))
    assert venv.get_attr("_randomization_scale") == [1.0, 1.0]


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


def test_train_cli_runs_sac(tmp_path):
    import subprocess, sys
    subprocess.run(
        [sys.executable, "-m", "shelfsort.train", "--algo", "sac", "--reward-mode", "dense", "--timesteps", "256",
         "--n-envs", "2", "--block-xy-range", "0.03", "--curriculum-steps", "128", "--normalize",
         "--out", str(tmp_path), "--run-name", "smoke"],
        check=True,
    )
    assert (tmp_path / "models" / "smoke.zip").exists()
    assert (tmp_path / "models" / "smoke_vecnormalize.pkl").exists()
