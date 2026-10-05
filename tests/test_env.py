import mujoco
import pytest
import numpy as np
from stable_baselines3.common.env_checker import check_env

from shelfsort.env import ShelfSortEnv


def test_action_space_is_4d_box_in_unit_range():
    env = ShelfSortEnv()

    assert env.action_space.shape == (4,)
    assert np.all(env.action_space.low == -1.0)
    assert np.all(env.action_space.high == 1.0)


def test_observation_space_is_21d_box():
    env = ShelfSortEnv()

    assert env.observation_space.shape == (21,)


@pytest.mark.parametrize("seed", [0, 1, 11])  # blue, green, red
def test_observation_ends_with_active_shelf_position(seed):
    """The policy is told where to go, not which colour it holds: a colour one-hot let each
    colour be learned as a separate mode, and green's collapsed to never grasping."""
    env = ShelfSortEnv()

    obs, _ = env.reset(seed=seed)

    np.testing.assert_allclose(obs[-3:], env.SHELF_ZONE_POS[env._active_category], atol=1e-6)


def test_reset_returns_observation_matching_space():
    env = ShelfSortEnv()

    obs, info = env.reset()

    assert obs.shape == env.observation_space.shape
    assert obs.dtype == env.observation_space.dtype
    assert isinstance(info, dict)


def test_reset_places_arm_joints_at_ready_pose():
    env = ShelfSortEnv()

    env.reset()

    arm_qpos = [env.data.joint(name).qpos[0] for name in env.ALL_JOINT_NAMES]
    expected = np.append(env.READY_ARM_QPOS, env.READY_GRIPPER_QPOS)
    np.testing.assert_allclose(arm_qpos, expected, atol=1e-8)


def test_ready_pose_points_gripper_straight_down():
    env = ShelfSortEnv()
    env.reset()

    approach_axis = env.data.site_xmat[env._ee_site_id].reshape(3, 3)[:, 0]

    np.testing.assert_allclose(approach_axis, [0.0, 0.0, -1.0], atol=0.05)


def test_ee_converges_to_a_reachable_target_without_oscillating():
    env = ShelfSortEnv()
    env.reset()
    target = np.array([0.16, -0.18, 0.06])

    for _ in range(60):
        error = target - env.data.site(env.EE_SITE_NAME).xpos
        env.step(np.array([*np.clip(error / 0.05, -0.5, 0.5), 1.0], dtype=np.float32))

    assert np.linalg.norm(target - env.data.site(env.EE_SITE_NAME).xpos) < 0.01


def test_scripted_pinch_grasps_and_lifts_the_block():
    """The gripper must be physically able to pick the block up, or no reward can teach it to."""
    env = ShelfSortEnv()
    env.reset(seed=0)
    block = env.data.body(env.BLOCK_BODY_NAMES[env._active_category]).xpos.copy()
    ee = lambda: env.data.site(env.EE_SITE_NAME).xpos.copy()

    def goto(target, steps=300):
        for _ in range(steps):
            error = target - ee()
            if np.linalg.norm(error) < 0.003:
                break
            env.step(np.array([*np.clip(error / 0.05, -0.5, 0.5), 1.0], dtype=np.float32))

    closing_axis = np.array([0.0, -1.0, 0.0])
    for _ in range(3):  # the closing axis rotates with the arm's pan, so re-aim after moving
        goto(block - 0.024 * closing_axis + [0.0, 0.0, 0.06 - block[2]])
        closing_axis = env.data.site_xmat[env._ee_site_id].reshape(3, 3)[:, 2].copy()
        closing_axis[2] = 0.0
        closing_axis /= np.linalg.norm(closing_axis)
    goto(np.array([*(block - 0.024 * closing_axis)[:2], 0.014]))

    close = np.interp(0.0, [-0.17453, 1.74533], [-1.0, 1.0])
    for _ in range(40):
        env.step(np.array([0.0, 0.0, 0.0, close], dtype=np.float32))
    for _ in range(15):
        _, _, _, _, info = env.step(np.array([0.0, 0.0, 0.5, close], dtype=np.float32))

    assert info["grasped"]
    assert env.data.body(env.BLOCK_BODY_NAMES[env._active_category]).xpos[2] > 0.04


@pytest.mark.parametrize("category", ShelfSortEnv.BLOCK_CATEGORIES)
def test_fully_closed_grasp_holds_the_block_while_carried_to_each_shelf(category):
    """The policy closes with action -1 and swings at full speed; the jaw hulls used to wedge
    the block out mid-carry, so only the nearest (red) zone was ever reachable."""
    env = ShelfSortEnv()
    env.reset(seed=0)
    block = env.data.body(env.BLOCK_BODY_NAMES[env._active_category]).xpos
    ee = lambda: env.data.site(env.EE_SITE_NAME).xpos.copy()

    def goto(target, gripper, speed, steps=300):
        for _ in range(steps):
            error = target - ee()
            if np.linalg.norm(error) < 0.003:
                break
            env.step(np.array([*np.clip(error / 0.05, -speed, speed), gripper], dtype=np.float32))

    start = block.copy()
    closing_axis = np.array([0.0, -1.0, 0.0])
    for _ in range(3):  # the closing axis rotates with the arm's pan, so re-aim after moving
        goto(start - 0.024 * closing_axis + [0.0, 0.0, 0.06 - start[2]], 1.0, 0.5)
        closing_axis = env.data.site_xmat[env._ee_site_id].reshape(3, 3)[:, 2].copy()
        closing_axis[2] = 0.0
        closing_axis /= np.linalg.norm(closing_axis)
    goto(np.array([*(start - 0.024 * closing_axis)[:2], 0.014]), 1.0, 0.5)
    for _ in range(40):
        env.step(np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32))

    goto(ee() + [0.0, 0.0, 0.06 - ee()[2]], -1.0, 1.0)
    shelf = env.SHELF_ZONE_POS[category]
    offset = ee() - block
    goto(np.array([*(shelf[:2] + offset[:2]), 0.06]), -1.0, 1.0)

    assert env._is_grasped()
    assert block[2] > 0.04
    assert np.linalg.norm(block[:2] - shelf[:2]) < env.SUCCESS_XY_TOLERANCE


def test_reset_places_active_block_at_supply_zone():
    env = ShelfSortEnv()

    env.reset()

    block_pos = env.data.body(env.BLOCK_BODY_NAMES[env._active_category]).xpos
    np.testing.assert_allclose(block_pos, env.SUPPLY_POS, atol=1e-6)


def test_step_returns_valid_gym_tuple():
    env = ShelfSortEnv()
    env.reset()

    obs, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))

    assert obs.shape == env.observation_space.shape
    assert obs.dtype == env.observation_space.dtype
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_invalid_reward_mode_rejected():
    with pytest.raises(ValueError):
        ShelfSortEnv(reward_mode="bogus")


def test_sparse_reward_is_zero_without_placement():
    env = ShelfSortEnv(reward_mode="sparse")
    env.reset(seed=0)

    rewards = [env.step(np.array([1.0, -1.0, 0.5, 1.0], dtype=np.float32))[1] for _ in range(5)]

    assert rewards == [0.0] * 5


def _teleport_block_to_shelf(env):
    cat = env._active_category
    qpos = env.data.joint(env.BLOCK_FREEJOINT_NAMES[cat]).qpos
    qpos[:3] = env.SHELF_ZONE_POS[cat] + np.array([0, 0, 0.016])
    mujoco.mj_forward(env.model, env.data)
    env._prev_reach_dist, env._prev_transport_dist = env._distances()  # teleport isn't shaping
    env._prev_height_potential = env._height_potential()


@pytest.mark.parametrize("mode", ["sparse", "dense"])
def test_placement_pays_bonus_and_terminates(mode):
    env = ShelfSortEnv(reward_mode=mode)
    env.reset(seed=0)
    _teleport_block_to_shelf(env)

    _, reward, terminated, _, info = env.step(np.zeros(4, dtype=np.float32))

    assert terminated and info["success"]
    assert reward == pytest.approx(1.0, abs=0.1)
    if mode == "sparse":
        assert reward == 1.0


def test_block_above_shelf_zone_is_not_success():
    """Carrying the block over the zone is not placing it there."""
    env = ShelfSortEnv()
    env.reset(seed=0)
    cat = env._active_category
    env.data.joint(env.BLOCK_FREEJOINT_NAMES[cat]).qpos[:3] = env.SHELF_ZONE_POS[cat] + np.array([0, 0, 0.06])
    mujoco.mj_forward(env.model, env.data)

    _, reward, terminated, _, info = env.step(np.zeros(4, dtype=np.float32))

    assert not info["success"] and not terminated
    assert reward == 0.0


def test_holding_the_block_still_is_not_rewarded():
    """A positive per-step reward while grasped made the policy dawdle with the block until
    timeout instead of placing it, since placing ends the episode."""
    env = ShelfSortEnv(reward_mode="dense")
    env.reset(seed=0)
    env._is_grasped = lambda: True
    env._ever_grasped = True

    _, reward, _, _, _ = env.step(np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32))

    assert reward <= 0.0


def test_dense_reward_pays_for_moving_toward_block():
    env = ShelfSortEnv(reward_mode="dense")
    env.reset(seed=0)
    block = env.data.body(env.BLOCK_BODY_NAMES[env._active_category]).xpos.copy()
    ee = env.data.site(env.EE_SITE_NAME).xpos.copy()
    direction = (block - ee) / np.linalg.norm(block - ee)

    total = sum(env.step(np.array([*direction, 1.0], dtype=np.float32))[1] for _ in range(3))

    assert total > 0.0


def test_info_reports_grasp_flags():
    env = ShelfSortEnv()
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
    assert info["grasped"] is False and info["ever_grasped"] is False


def test_step_moves_end_effector_toward_commanded_direction():
    env = ShelfSortEnv()
    env.reset()
    ee_pos_before = np.array(env.data.site(env.EE_SITE_NAME).xpos)

    up_action = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    for _ in range(20):
        env.step(up_action)

    ee_pos_after = np.array(env.data.site(env.EE_SITE_NAME).xpos)

    assert ee_pos_after[2] > ee_pos_before[2] + 0.02


def test_episode_truncates_after_max_steps_without_success():
    env = ShelfSortEnv()
    env.reset()

    no_op = np.zeros(4, dtype=np.float32)
    for _ in range(env.MAX_EPISODE_STEPS - 1):
        _, _, terminated, truncated, _ = env.step(no_op)
        assert not truncated

    _, _, terminated, truncated, _ = env.step(no_op)

    assert truncated
    assert not terminated


def test_env_passes_sb3_check_env():
    env = ShelfSortEnv()

    check_env(env, warn=True)


def test_info_reports_active_category():
    env = ShelfSortEnv()
    env.reset(seed=0)

    _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))

    assert info["category"] == env._active_category


def test_dense_reward_pays_for_lifting_the_block_and_sparse_does_not():
    rewards = {}
    for mode in ("sparse", "dense"):
        env = ShelfSortEnv(reward_mode=mode)
        env.reset(seed=0)
        qpos = env.data.joint(env.BLOCK_FREEJOINT_NAMES[env._active_category]).qpos
        qpos[2] += 0.03
        mujoco.mj_forward(env.model, env.data)
        rewards[mode] = env.step(np.zeros(4, dtype=np.float32))[1]

    assert rewards["sparse"] == 0.0
    assert rewards["dense"] > 0.1


@pytest.mark.parametrize("over_zone", [True, False])
def test_lowering_the_block_pays_only_over_its_shelf_zone(over_zone):
    """Lowering used to cost the lift reward everywhere, so the policy carried the block to the
    zone and held it there instead of putting it down."""
    env = ShelfSortEnv(reward_mode="dense")
    env.reset(seed=0)
    cat = env._active_category
    qpos = env.data.joint(env.BLOCK_FREEJOINT_NAMES[cat]).qpos
    xy = env.SHELF_ZONE_POS[cat][:2] if over_zone else env.SUPPLY_POS[:2]
    qpos[:3] = [*xy, env.BLOCK_REST_Z + 0.03]
    mujoco.mj_forward(env.model, env.data)
    env._prev_reach_dist, env._prev_transport_dist = env._distances()
    env._prev_height_potential = env._height_potential()
    qpos[2] = env.BLOCK_REST_Z + 0.01
    mujoco.mj_forward(env.model, env.data)

    _, reward, _, _, _ = env.step(np.zeros(4, dtype=np.float32))

    assert (reward > 0.0) == over_zone


def test_lift_term_is_capped():
    env = ShelfSortEnv(reward_mode="dense")
    env.reset(seed=0)
    qpos = env.data.joint(env.BLOCK_FREEJOINT_NAMES[env._active_category]).qpos
    qpos[2] += 0.5
    mujoco.mj_forward(env.model, env.data)

    assert env._lift() == env.LIFT_CAP
