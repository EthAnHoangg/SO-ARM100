from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

DEFAULT_SCENE_PATH = Path(__file__).resolve().parent.parent / "Simulation" / "SO101" / "scene.xml"


class ShelfSortEnv(gym.Env):
    ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    GRIPPER_JOINT_NAME = "gripper"
    ALL_JOINT_NAMES = ARM_JOINT_NAMES + [GRIPPER_JOINT_NAME]
    EE_SITE_NAME = "gripperframe"

    # Ready pose: gripper pointing straight down at (0.18, -0.10, 0.09), clear of the blocks. All-zero
    # joints leave the gripper horizontal, and the controller's first moves would swing it down.
    READY_ARM_QPOS = np.array([0.616, -0.115, 0.05, 1.637, 0.0])
    READY_GRIPPER_QPOS = 1.0

    BLOCK_CATEGORIES = ("red", "green", "blue")
    BLOCK_BODY_NAMES = {c: f"block_{c}" for c in BLOCK_CATEGORIES}
    BLOCK_FREEJOINT_NAMES = {c: f"block_{c}_freejoint" for c in BLOCK_CATEGORIES}

    SUPPLY_POS = np.array([0.16, -0.18, 0.016])
    STAGING_POSITIONS = [
        np.array([0.05, -0.30, 0.016]),
        np.array([0.05, -0.36, 0.016]),
        np.array([0.05, -0.42, 0.016]),
    ]
    SHELF_ZONE_POS = {
        "red": np.array([0.24, -0.10, 0.002]),
        "green": np.array([0.27, 0.02, 0.002]),
        "blue": np.array([0.24, 0.14, 0.002]),
    }

    EE_STEP_SIZE = 0.02  # metres of EE travel commanded by an action of magnitude 1
    N_SUBSTEPS = 5  # physics steps per env step, giving the position actuators time to respond
    MAX_EPISODE_STEPS = 200
    SUCCESS_XY_TOLERANCE = 0.03  # metres, block-to-shelf-zone planar distance counted as "placed"

    REWARD_MODES = ("sparse", "dense")
    PLACEMENT_REWARD = 1.0  # identical in both modes so shaping is the only difference
    REACH_WEIGHT = 2.0  # per metre of EE-to-block distance closed (kept well under the placement bonus)
    TRANSPORT_WEIGHT = 5.0  # per metre of block-to-shelf distance closed while held
    GRASP_BONUS = 0.5  # paid once per episode, on first grasp
    STEP_PENALTY = 0.001

    APPROACH_DIR = np.array([0.0, 0.0, -1.0])  # desired world direction of the gripper approach axis
    ORIENTATION_WEIGHT_LOW = 0.4  # near the table, where the pinch happens
    ORIENTATION_WEIGHT_HIGH = 0.1  # when raised: wrist_flex runs out of range if forced to stay level
    ORIENTATION_BLEND_Z = (0.04, 0.10)  # EE heights over which the weight ramps from LOW to HIGH
    MAX_TILT_STEP = 0.2  # rad of tilt correction per env step
    DLS_DAMPING = 1e-3
    N_CONTROLLED_JOINTS = 4  # everything but wrist_roll

    FIXED_JAW_BODY_NAME = "gripper"
    MOVING_JAW_BODY_NAME = "moving_jaw_so101_v1"

    def __init__(self, scene_path=DEFAULT_SCENE_PATH, reward_mode="sparse"):
        if reward_mode not in self.REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {self.REWARD_MODES}, got {reward_mode!r}")
        self.reward_mode = reward_mode
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32)

        self._active_category = self.BLOCK_CATEGORIES[0]
        self._arm_ctrl_target = np.zeros(len(self.ARM_JOINT_NAMES))
        self._elapsed_steps = 0
        self._prev_reach_dist = 0.0
        self._prev_transport_dist = 0.0
        self._ever_grasped = False

        self._fixed_jaw_body_id = self.model.body(self.FIXED_JAW_BODY_NAME).id
        self._moving_jaw_body_id = self.model.body(self.MOVING_JAW_BODY_NAME).id
        self._ee_site_id = self.model.site(self.EE_SITE_NAME).id
        self._arm_dof_adr = [self.model.joint(name).dofadr[0] for name in self.ARM_JOINT_NAMES]
        self._arm_joint_range = np.array([self.model.joint(name).range for name in self.ARM_JOINT_NAMES])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self._active_category = self.BLOCK_CATEGORIES[self.np_random.integers(len(self.BLOCK_CATEGORIES))]

        ready_qpos = np.append(self.READY_ARM_QPOS, self.READY_GRIPPER_QPOS)
        for name, qpos in zip(self.ALL_JOINT_NAMES, ready_qpos):
            joint = self.data.joint(name)
            joint.qpos[0] = qpos
            joint.qvel[0] = 0.0
            self.data.actuator(name).ctrl[0] = qpos

        staging = iter(self.STAGING_POSITIONS)
        for category in self.BLOCK_CATEGORIES:
            freejoint = self.data.joint(self.BLOCK_FREEJOINT_NAMES[category])
            pos = self.SUPPLY_POS if category == self._active_category else next(staging)
            freejoint.qpos[:3] = pos
            freejoint.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            freejoint.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

        self._arm_ctrl_target = self.READY_ARM_QPOS.copy()
        self._elapsed_steps = 0
        self._ever_grasped = False
        self._prev_reach_dist, self._prev_transport_dist = self._distances()

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        delta_ee = action[:3] * self.EE_STEP_SIZE
        gripper_action = action[3]

        # Position task (3) plus an orientation task (3) that keeps the gripper's approach axis
        # pointing straight down, which is what a top-down pinch grasp needs. The arm has only 5
        # joints, so the two tasks are blended with damped least squares; the rotation about the
        # approach axis is left to wrist_roll. Anchoring the target on the measured joint angles
        # (not the previous target) stops integrator wind-up.
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._ee_site_id)
        approach = self.data.site_xmat[self._ee_site_id].reshape(3, 3)[:, 0]
        tilt_error = np.clip(np.cross(approach, self.APPROACH_DIR), -self.MAX_TILT_STEP, self.MAX_TILT_STEP)

        # wrist_roll is held at its neutral angle: left free it winds up to its limit.
        ee_z = self.data.site_xpos[self._ee_site_id][2]
        blend = np.clip((ee_z - self.ORIENTATION_BLEND_Z[0]) / np.diff(self.ORIENTATION_BLEND_Z)[0], 0.0, 1.0)
        weight = self.ORIENTATION_WEIGHT_LOW + blend * (self.ORIENTATION_WEIGHT_HIGH - self.ORIENTATION_WEIGHT_LOW)
        dofs = self._arm_dof_adr[: self.N_CONTROLLED_JOINTS]
        jac = np.vstack([jacp[:, dofs], weight * jacr[:, dofs]])
        task = np.concatenate([delta_ee, weight * tilt_error])
        q_now = np.array([self.data.qpos[adr] for adr in self._arm_dof_adr])
        q_now[self.N_CONTROLLED_JOINTS :] = 0.0
        lo, hi = self._arm_joint_range[:, 0], self._arm_joint_range[:, 1]
        n = self.N_CONTROLLED_JOINTS

        # A joint that would be pushed past its limit is pinned there and removed from the solve,
        # so the remaining joints make up the motion instead of the error leaking into x/y drift.
        delta_q = np.zeros(len(self.ARM_JOINT_NAMES))
        free = np.ones(n, dtype=bool)
        for _ in range(n):
            residual = task - jac[:, ~free] @ delta_q[:n][~free] if (~free).any() else task
            jf = jac[:, free]
            delta_q[:n][free] = jf.T @ np.linalg.solve(jf @ jf.T + self.DLS_DAMPING * np.eye(6), residual)
            proposed = q_now[:n] + delta_q[:n]
            over = free & ((proposed < lo[:n]) | (proposed > hi[:n]))
            if not over.any():
                break
            delta_q[:n][over] = np.clip(proposed[over], lo[:n][over], hi[:n][over]) - q_now[:n][over]
            free &= ~over
        self._arm_ctrl_target = np.clip(q_now + delta_q, lo, hi)

        for name, target in zip(self.ARM_JOINT_NAMES, self._arm_ctrl_target):
            self.data.actuator(name).ctrl[0] = target

        gripper_range = self.model.actuator(self.GRIPPER_JOINT_NAME).ctrlrange
        gripper_target = np.interp(gripper_action, [-1.0, 1.0], gripper_range)
        self.data.actuator(self.GRIPPER_JOINT_NAME).ctrl[0] = gripper_target

        for _ in range(self.N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        self._elapsed_steps += 1

        reach_dist, transport_dist = self._distances()
        block_pos = self.data.body(self.BLOCK_BODY_NAMES[self._active_category]).xpos
        target_pos = self.SHELF_ZONE_POS[self._active_category]
        xy_distance = np.linalg.norm(block_pos[:2] - target_pos[:2])
        success = bool(xy_distance < self.SUCCESS_XY_TOLERANCE)
        grasped = self._is_grasped()

        reward = self.PLACEMENT_REWARD if success else 0.0
        if self.reward_mode == "dense":
            reward += self.REACH_WEIGHT * (self._prev_reach_dist - reach_dist)
            if grasped:
                if not self._ever_grasped:
                    reward += self.GRASP_BONUS
                reward += self.TRANSPORT_WEIGHT * (self._prev_transport_dist - transport_dist)
            reward -= self.STEP_PENALTY
        self._prev_reach_dist, self._prev_transport_dist = reach_dist, transport_dist
        self._ever_grasped = self._ever_grasped or grasped

        terminated = success
        truncated = self._elapsed_steps >= self.MAX_EPISODE_STEPS
        info = {"success": success, "grasped": grasped, "ever_grasped": self._ever_grasped}

        return self._get_obs(), float(reward), terminated, truncated, info

    def _distances(self):
        block_pos = self.data.body(self.BLOCK_BODY_NAMES[self._active_category]).xpos
        ee_pos = self.data.site(self.EE_SITE_NAME).xpos
        target_pos = self.SHELF_ZONE_POS[self._active_category]
        return float(np.linalg.norm(ee_pos - block_pos)), float(np.linalg.norm(block_pos - target_pos))

    def _is_grasped(self):
        """True when the active block touches both the fixed and the moving jaw."""
        block_body_id = self.model.body(self.BLOCK_BODY_NAMES[self._active_category]).id
        touching = set()
        for contact in self.data.contact[: self.data.ncon]:
            bodies = {self.model.geom_bodyid[contact.geom1], self.model.geom_bodyid[contact.geom2]}
            if block_body_id in bodies:
                touching |= bodies - {block_body_id}
        return self._fixed_jaw_body_id in touching and self._moving_jaw_body_id in touching

    def _get_obs(self):
        qpos = np.array([self.data.joint(name).qpos[0] for name in self.ALL_JOINT_NAMES], dtype=np.float32)
        qvel = np.array([self.data.joint(name).qvel[0] for name in self.ALL_JOINT_NAMES], dtype=np.float32)
        ee_pos = np.array(self.data.site(self.EE_SITE_NAME).xpos, dtype=np.float32)
        block_pos = np.array(self.data.body(self.BLOCK_BODY_NAMES[self._active_category]).xpos, dtype=np.float32)

        category_onehot = np.zeros(len(self.BLOCK_CATEGORIES), dtype=np.float32)
        category_onehot[self.BLOCK_CATEGORIES.index(self._active_category)] = 1.0

        shelf_positions = np.concatenate([self.SHELF_ZONE_POS[c] for c in self.BLOCK_CATEGORIES]).astype(np.float32)

        return np.concatenate([qpos, qvel, ee_pos, block_pos, category_onehot, shelf_positions])
