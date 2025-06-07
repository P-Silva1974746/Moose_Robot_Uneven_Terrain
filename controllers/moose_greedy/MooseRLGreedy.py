import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math
from moose_greedy import MooseNavigator

class MooseRLGymEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30} # optional

    def __init__(self):
        super(MooseRLGymEnv, self).__init__()
        self.navigator = MooseNavigator()

        # Define Observation Space
        low_obs = np.array([-np.inf, -np.inf, -np.pi, 0.0, 0.0, 0.0, -np.pi, -np.pi, -np.pi, 0.0, 0.0], dtype=np.float32)
        high_obs = np.array([np.inf, np.inf, np.pi, self.navigator.x_dim * self.navigator.x_spacing, self.navigator.y_dim * self.navigator.y_spacing, np.inf, np.pi, np.pi, np.pi, np.inf, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)


        self.action_space = spaces.Box(low=np.array([-self.navigator.max_speed, -self.navigator.max_speed]),
                                       high=np.array([self.navigator.max_speed, self.navigator.max_speed]),
                                       dtype=np.float32)

        # Internal state for stuck detection
        self.last_robot_pos = None
        self.stuck_check_interval = 500 # steps


    def get_greedy_action(self, obs):
        # 'obs' array numpy returned by _get_obs()

        robot_pos = obs[0:3]
        robot_yaw = obs[2]  # only yaw
        goal_pos = obs[3:5]  # only goal_x, goal_y

        left_speed, right_speed = self.navigator.get_greedy_motor_commands(robot_pos, robot_yaw, goal_pos)

        return [left_speed, right_speed]


    def _get_obs(self):
        pos = self.navigator.gps.getValues()
        heading = self.navigator.get_heading()
        distance = self.navigator.distance_to_goal(pos)
        angle = self.navigator.angle_to_goal(pos)
        pitch, roll, _ = self.navigator.imu.getRollPitchYaw()

        print(f"[OBS] Posição GPS (pos): {pos}")

        if np.isnan(pos).any():
            print("[OBS] Valores NaN detetados na posição GPS!")

        current_grid_x = int(pos[0] / self.navigator.x_spacing)
        current_grid_y = int(pos[1] / self.navigator.y_spacing)
        slope_mag, normal_vec = self.navigator.get_slope_value(current_grid_x, current_grid_y, scale=10)

        return np.array([pos[0], pos[1], heading, self.navigator.goal[0], self.navigator.goal[1],
                         distance, angle, pitch, roll, slope_mag, normal_vec[2]], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        random_x = random.uniform(20.0, 180.0)
        random_y = random.uniform(20.0, 180.0)

        height = self.navigator.get_terrain_height(random_x, random_y)
        robot_z = height + 0.5

        self.navigator.robot_node.getField("translation").setSFVec3f([random_x, random_y, robot_z])
        self.navigator.robot_node.getField("rotation").setSFRotation([0, 0, 1, random.uniform(-math.pi, math.pi)])

        goal_x, goal_y = self.navigator.set_goal_position()
        print(f"Goal position: ({goal_x:.2f}, {goal_y:.2f})")

        for _ in range(5):
            self.navigator.robot.step(self.navigator.timestep)
        observation = self._get_obs()

        self.last_robot_pos = self.navigator.gps.getValues()
        self.steps_since_last_check = 0

        info = {}

        return observation, info


    def step(self, action):
        # Apply action to Webots motors
        left_speed, right_speed = action[0], action[1]
        self.navigator.set_motor_velocity(left_speed, right_speed)

        # Advance simulation
        step_result = self.navigator.robot.step(self.navigator.timestep)

        if step_result == -1:
            return self._get_obs(), 0, True, False, {}  # Episode interrupted


        # new observation
        observation = self._get_obs()

        current_pos = self.navigator.gps.getValues()
        distance_to_goal = self.navigator.distance_to_goal(current_pos)
        pitch, roll, _ = self.navigator.imu.getRollPitchYaw()

        reward = 0.0
        terminated = False
        truncated = False

        # Reward: Getting closer to goal
        reward += (self.navigator.distance_to_goal(self.last_robot_pos) - distance_to_goal) * 10.0

        # Penalties:
        reward -= 0.01

        pitch_threshold = 0.25
        roll_threshold = 0.25
        if abs(pitch) > pitch_threshold or abs(roll) > roll_threshold:
            reward -= 10.0  # penalty for instability
            terminated = True

        # Check for stuck
        self.steps_since_last_check += 1
        if self.steps_since_last_check >= self.stuck_check_interval:
            if self.navigator.is_robot_stuck(current_pos, self.last_robot_pos):
                reward -= 100.0  # penalty for being stuck
                terminated = True
            self.last_robot_pos = current_pos
            self.steps_since_last_check = 0

        # Success condition
        if distance_to_goal < 0.2:
            reward += 1000.0  # big reawrd for reaching the goal
            terminated = True
            print(
                f"[RL_ENV] Reward: {reward:.4f}, Terminated: {terminated}")

        info = {}

        return observation, reward, terminated, truncated, info

    def close(self):
        pass