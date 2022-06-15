import time, random
from typing import Optional, Union
import gym
from gym import spaces
import numpy as np
from gym_dogfight3d.envs.uav import UAV
from gym_dogfight3d.envs.utils import *
import math
import line_profiler

class DogFightEnv(gym.Env):
    """A 3D Air combat environment for OpenAI gym"""
    visualization = None

    def __init__(self):
        super(DogFightEnv, self).__init__()
        self.current_step = 0
        self.uav1 = UAV()
        self.uav2 = UAV()
        self.uav_list = [self.uav1, self.uav2]
        self.target_distance = 0
        self.target_angle = 0
        self.continuous = True
        self.dt = 0.0625
        self.info = {"attack": 0, "be_attacked": 0, "fallInWater": 0} # fallInWater,0:don't fall, 1:uav1 fall, 2: uav2 fall
        if self.continuous:
            # thrust, pitch, roll, yaw
            # throttle(油门), elevator(升降), aileron(副翼), rudder(方向舵)
            self.action_space = spaces.Box(-1.0, 1.0, np.array([4,]), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(-1.0, 1.0, np.array([17,]), dtype=np.float64)

    def _next_obs(self, uav1, uav2):
        obs = np.zeros(self.observation_space.shape[0])
        # ownship
        obs[0] = uav1.position[0] / 2000
        obs[1] = uav1.position[2] / 2000
        obs[2] = uav1.position[1] / 5000
        obs[3] = uav1.linear_speed / 300
        obs[4] = uav1.linear_acceleration / 10
        obs[5] = uav1.health_level
        obs[6] = uav1.cap / 360
        obs[7] = uav1.pitch_attitude / 90
        obs[8] = uav1.roll_attitude / 90
        obs[9] = uav1.thrust_level
        # opponent
        obs[10] = uav2.position[0] / 2000
        obs[11] = uav2.position[2] / 2000
        obs[12] = uav2.position[1] / 5000
        obs[13] = uav2.linear_speed / 300
        obs[14] = uav2.cap / 360
        obs[15] = uav2.pitch_attitude / 90
        obs[16] = uav2.roll_attitude / 90
        return obs

    def _take_action(self, uav, action):
        throttle, elevator, aileron, rudder = action
        uav.set_thrust_level(uav.thrust_level + throttle * 0.1)
        uav.set_pitch_level(elevator)
        uav.set_roll_level(aileron)
        uav.set_yaw_level(rudder)

    def attack(self, uav1: UAV, uav2: UAV):
        attack_reward = 0
        # uav1 attack uav2
        self.target_distance = np.linalg.norm(uav2.position - uav1.position)
        uav1_dir = uav1.direction
        tar_dir = normalize(uav2.position - uav1.position)
        self.target_angle = math.degrees(math.acos(max(-1, min(1, np.dot(uav1_dir, tar_dir)))))
        if self.target_angle < uav1.attack_angle and uav1.attack_range[0] < self.target_distance < uav1.attack_range[1]:
            uav2.hit(0.05)
            attack_reward = 0.05
        return attack_reward

    def track(self):
        return (30 - self.target_angle) * 0.00001

    def reset(self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        init_thrust_level = 0.7
        init_linear_speed = 800 / 3.6
        pos_range = 1500
        init_postion1 = np.array([random.randint(-pos_range, pos_range),
                                  random.randint(3000, 4000),
                                  random.randint(-pos_range, pos_range)], dtype=np.float32)
        init_rotation1 = np.array([math.radians(random.randint(-180, 180)), 0, 0], dtype=np.float32)
        init_postion2 = np.array([random.randint(-pos_range, pos_range),
                                  random.randint(3000, 4000),
                                  random.randint(-pos_range, pos_range)], dtype=np.float32)
        init_rotation2 = np.array([math.radians(random.randint(-180, 180)), 0, 0], dtype=np.float32)
        self.uav1.reset(init_thrust_level, init_linear_speed, init_postion1, init_rotation1)
        self.uav2.reset(init_thrust_level, init_linear_speed, init_postion2, init_rotation2)
        self.current_step = 0
        self.info = {"attack": 0, "be_attacked": 0}
        if not return_info:
            return self._next_obs(self.uav1, self.uav2)
        else:
            return self._next_obs(self.uav1, self.uav2), {}

    def step(self, action):
        self.current_step += 1
        fall_reward = 0
        self._take_action(self.uav1, action)
        opponent_action = self.action_space.sample()
        self._take_action(self.uav2, opponent_action)
        self.uav1.update_kinetics(self.dt)
        self.uav2.update_kinetics(self.dt)
        attack_reward = self.attack(self.uav1, self.uav2)
        if attack_reward != 0: self.info["attack"] += 1
        be_attacked_reward = -self.attack(self.uav2, self.uav1)
        if be_attacked_reward != 0: self.info["be_attacked"] += 1
        track_reward = self.track()
        done = self.uav1.wreck or self.uav2.wreck
        if done:
            if self.uav1.fallInWater:
                self.info["fallInWater"] = 1
                fall_reward = -2
            if self.uav2.fallInWater: self.info["fallInWater"] = 2
        reward = attack_reward + be_attacked_reward + track_reward + fall_reward
        return self._next_obs(self.uav1, self.uav2), reward, done, self.info

    def render(self):
        pass

    def close(self):
        pass


def main():
    env = DogFightEnv()
    init = env.reset()
    start = time.time()
    while True:
        action = env.action_space.sample()
        next_obs, r, done, info = env.step(action)
        if done:
            print(info, env.current_step)
            break
    print(time.time() - start)

if __name__ == '__main__':
    main()