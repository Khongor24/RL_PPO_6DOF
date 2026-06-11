import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco


class MyRobotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path("my_robot.xml")
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.viewer = None

        # RL action range stays -1 to +1
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.model.nu,),
            dtype=np.float32,
        )

        # Observation = joint positions + joint velocities + tip position + target
        obs_size = self.model.nq + self.model.nv + 3 + 3

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )

        # Must match target position in my_robot.xml:
        # <body name="target" pos="0.6 0.1 0.1">
        self.target = np.array([0.6, 0.1, 0.1], dtype=np.float32)

        # Episode limit
        self.max_steps = 300
        self.current_step = 0

        # Make control less sensitive
        # Action from PPO is still between -1 and 1,
        # but only 25% is sent to the motors.
        self.action_scale = 0.25

    def _get_obs(self):
        obs = np.concatenate([
            self.data.qpos,
            self.data.qvel,
            self._get_tip_position(),
            self.target,
        ])

        return obs.astype(np.float32)

    def _get_tip_position(self):
        tip_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "tip",
        )

        return self.data.xpos[tip_id].copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0

        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:] = self.np_random.uniform(
            low=-0.1,
            high=0.1,
            size=self.model.nq,
        )

        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

        info = {
            "target": self.target.copy(),
            "tip_position": self._get_tip_position(),
        }

        return self._get_obs(), info

    def step(self, action):
        self.current_step += 1

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Scale action before sending to MuJoCo
        scaled_action = action * self.action_scale

        self.data.ctrl[:] = scaled_action

        # Run several MuJoCo physics steps per one RL step
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        tip_pos = self._get_tip_position()
        distance = np.linalg.norm(tip_pos - self.target)

        # Main reward: negative distance (penalize being far)
        reward = -distance
        
        # For reward shaping: approximate max distance at reset
        self.max_distance = 2.0

        # Penalize large original actions
        reward -= 0.01 * float(np.sum(np.square(action)))

        # Penalize high joint velocity to reduce jitter
        reward -= 0.001 * float(np.sum(np.square(self.data.qvel)))

        # Success bonus
        if distance < 0.05:
            reward += 5.0

        terminated = bool(distance < 0.05)
        truncated = bool(self.current_step >= self.max_steps)

        info = {
            "distance": float(distance),
            "target": self.target.copy(),
            "tip_position": tip_pos.copy(),
            "raw_action": action.copy(),
            "scaled_action": scaled_action.copy(),
            "is_success": bool(distance < 0.05),
        }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return

        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None