import time, random
from typing import Optional, Union
import gym
from gym import spaces
import numpy as np
from gym_dogfight3d.envs.uav import UAV
from gym_dogfight3d.envs.utils import *
import math, transforms3d
import matplotlib.pyplot as plt
from matplotlib import cm
import line_profiler

class DogFightEnv(gym.Env):
    """A 3D Air combat environment for OpenAI gym"""

    def __init__(self, op_policy='self-play'):
        super(DogFightEnv, self).__init__()
        self.current_step = 0
        self.uav1 = UAV()
        self.uav2 = UAV()
        self.uav_list = [self.uav1, self.uav2]
        self.target_distance = 0
        self.target_angle = 0
        self.continuous = True
        self.dt = 0.0625
        self.ax = None
        self.op_policy = op_policy
        self.info = {"attack": 0, "be_attacked": 0, "fallInWater": 0} # fallInWater,0:don't fall, 1:uav1 fall, 2: uav2 fall
        if self.continuous:
            # thrust, pitch, roll, yaw
            # throttle(油门), elevator(升降), aileron(副翼), rudder(方向舵)
            self.action_space = spaces.Box(-1.0, 1.0, np.array([4, ]), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(-11.0, 11.0, np.array([20, ]), dtype=np.float64)

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
        # relative
        obs[17] = (uav1.position[0] - uav2.position[0]) / 500
        obs[18] = (uav1.position[2] - uav2.position[2]) / 500
        obs[19] = (uav1.position[1] - uav2.position[1]) / 500
        return obs

    def _take_action(self, uav, action):
        throttle, elevator, aileron, rudder = action
        uav.set_thrust_level(uav.thrust_level + throttle * 0.01)
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

    def get_shaped_reward(self):
        track_reward = (10 - self.target_angle) * 1e-6
        distance_reward = (800 - self.target_distance) * 1e-7
        height_reward = (self.uav1.position[1] - 500) * 1e-5 if self.uav1.position[1] < 500 else 0
        return track_reward + distance_reward + height_reward


    def reset(self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.ax = None
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
        if self.op_policy == 'self-play':
            if not return_info:
                return [self._next_obs(self.uav1, self.uav2), self._next_obs(self.uav2, self.uav1)]
            else:
                return [self._next_obs(self.uav1, self.uav2), self._next_obs(self.uav2, self.uav1)], {}
        else:
            if not return_info:
                return self._next_obs(self.uav1, self.uav2)
            else:
                return self._next_obs(self.uav1, self.uav2), {}

    def step(self, action):
        self.current_step += 1
        fall_reward = 0
        if self.op_policy == 'self-play':
            uav1_action, uav2_action = action
        else:
            uav1_action, uav2_action = action, self.action_space.sample()
        self._take_action(self.uav1, uav1_action)
        self._take_action(self.uav2, uav2_action)
        self.uav1.update_kinetics(self.dt)
        self.uav2.update_kinetics(self.dt)
        be_attacked_reward = -self.attack(self.uav2, self.uav1)
        if be_attacked_reward != 0: self.info["be_attacked"] += 1
        attack_reward = self.attack(self.uav1, self.uav2)
        if attack_reward != 0: self.info["attack"] += 1
        done = self.uav1.wreck or self.uav2.wreck
        if done:
            if self.uav1.fallInWater:
                self.info["fallInWater"] = 1
                fall_reward = -2
            if self.uav2.fallInWater: self.info["fallInWater"] = 2
        reward = attack_reward + be_attacked_reward + fall_reward + self.get_shaped_reward()
        if self.op_policy == 'self-play':
            return [self._next_obs(self.uav1, self.uav2), self._next_obs(self.uav2, self.uav1)], reward, done, self.info
        return self._next_obs(self.uav1, self.uav2), reward, done, self.info

    # inefficient renderer
    # def _render(self):
    #     if self.ax is None:
    #         self.ax = plt.axes(projection='3d')
    #         self.ax.set_title('dogfight3d')
    #         # axes range
    #         self.ax.set_xlim([-2000, 2000])
    #         self.ax.set_xlabel('X')
    #         self.ax.set_xticks(range(-2000, 2001, 1000))
    #         self.ax.set_ylim([0, 5000])
    #         self.ax.set_ylabel('Y')
    #         self.ax.set_yticks(range(0, 5001, 1000))
    #         self.ax.set_zlim([-2000, 2000])
    #         self.ax.set_zlabel('Z')
    #         self.ax.set_zticks(range(-2000, 2001, 1000))
    #         self.uav1_pos = [[], [], []]
    #         self.uav2_pos = [[], [], []]
    #         plt.ion()
    #     self.uav1_pos[0].append(self.uav1.position[0])
    #     self.uav1_pos[1].append(self.uav1.position[1])
    #     self.uav1_pos[2].append(self.uav1.position[2])
    #     self.uav2_pos[0].append(self.uav2.position[0])
    #     self.uav2_pos[1].append(self.uav2.position[1])
    #     self.uav2_pos[2].append(self.uav2.position[2])
    #     self.tmp1, = plt.plot(self.uav1_pos[0], self.uav1_pos[1], self.uav1_pos[2], 'red')
    #     self.tmp2, = plt.plot(self.uav2_pos[0], self.uav2_pos[1], self.uav2_pos[2], 'blue')
    #
    #     font = {'family': 'serif',
    #             'color': 'darkred',
    #             'weight': 'normal',
    #             'size': 16,
    #             }
    #     plt.legend(labels=[f"{self.uav1.health_level:.3f}",f"{self.uav2.health_level:.3f}"],loc="upper left", bbox_to_anchor=(-0.1, 0, 0, 1.1))
    #     plt.pause(0.00001)

    def _render(self):
        if self.ax is None:
            self.ax = plt.axes(projection='3d')
            self.ax.set_title('dogfight3d')
            # axes range
            self.ax.set_xlim([-3000, 3000])
            self.ax.set_ylim([-3000, 3000])
            self.ax.set_zlim([0, 5000])

            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')

            self.ax.set_xticks(range(-3000, 3001, 1000))
            self.ax.set_zticks(range(0, 5001, 1000))
            self.ax.set_yticks(range(-3000, 3001, 1000))
            self.uav1_pos = [[], [], []]
            self.uav2_pos = [[], [], []]
            self.line1, = plt.plot([0,0,0], [0,0,0], 'r')
            self.line2, = plt.plot([0,0,0], [0,0,0], 'b')
            plt.ion()
        self.uav1_pos[0].append(self.uav1.position[0])
        self.uav1_pos[1].append(self.uav1.position[1])
        self.uav1_pos[2].append(self.uav1.position[2])
        self.uav2_pos[0].append(self.uav2.position[0])
        self.uav2_pos[1].append(self.uav2.position[1])
        self.uav2_pos[2].append(self.uav2.position[2])
        font = {'family': 'serif',
                'color': 'darkred',
                'weight': 'normal',
                'size': 16,
                }
        plt.legend(labels=[f"{self.uav1.health_level:.3f}", f"{self.uav2.health_level:.3f}"], loc="upper left",
                   bbox_to_anchor=(-0.1, 0, 0, 1.1))
        # trajectory
        self.line1.set_xdata(self.uav1_pos[0])
        self.line1.set_ydata(self.uav1_pos[2])
        self.line1.set_3d_properties(self.uav1_pos[1])
        self.line2.set_xdata(self.uav2_pos[0])
        self.line2.set_ydata(self.uav2_pos[2])
        self.line2.set_3d_properties(self.uav2_pos[1])

        # attack range
        p = np.linspace(0, 2 * np.pi, 20)
        r = np.linspace(0, 139, 20)
        R, P = np.meshgrid(r, p)
        x = R * np.cos(P)
        y = R * np.sin(P)
        z = np.sqrt(x ** 2 + y ** 2) / math.radians(10)
        x, y, z = x.flatten(), y.flatten(), z.flatten()
        # rotation
        # mat1 = transforms3d.euler.euler2mat(self.uav1.rotation[0], self.uav1.rotation[1], self.uav1.rotation[2], 'ryxz')
        mat1 = transforms3d.euler.euler2mat(self.uav1.rotation[0], self.uav1.rotation[1],  self.uav1.rotation[2], 'ryxz')
        # print(self.uav1.rotation[0], self.uav1.rotation[1], self.uav1.pitch_attitude)
        x1, z1, y1 = np.dot(mat1, np.stack([x, y, z], 0)) + self.uav1.position.reshape(3, 1)
        x1, y1, z1 = x1.reshape((20, -1)), y1.reshape((20, -1)), z1.reshape((20, -1))
        cone1 = self.ax.plot_surface(x1, y1, z1, color="crimson")
        mat2 = transforms3d.euler.euler2mat(self.uav2.rotation[0], self.uav2.rotation[1], self.uav2.rotation[2], 'ryxz')
        x2, z2, y2 = np.dot(mat2, np.stack([x, y, z], 0)) + self.uav2.position.reshape(3, 1)
        x2, y2, z2 = x2.reshape((20, -1)), y2.reshape((20, -1)), z2.reshape((20, -1))
        cone2 = self.ax.plot_surface(x2, y2, z2, color="royalblue")

        self.ax.set_xlim([-3000, 3000])
        self.ax.set_zlim([0, 5000])
        self.ax.set_ylim([-3000, 3000])
        plt.pause(0.00001)
        cone1.remove()
        cone2.remove()

    def render(self, mode='live'):
        assert mode in ["live", "replay"], "Invalid mode, must be either \"live\" or \"replay\""
        if mode == 'replay':
            pass
        elif mode == 'live':
            self._render()

    def close(self):
        pass


def test():
    env = DogFightEnv()
    init = env.reset()
    start = time.time()
    print(start)
    steps = 0
    # actions = []
    # with open('action.txt', 'r') as f:
    #     for line in f:
    #         actions.append(list(map(float, line.strip().split())))
    while True:
        env.render('live')
        action = env.action_space.sample()
        # action = actions[steps]
        # print(action)
        steps += 1
        next_obs, r, done, info = env.step(action)
        # print(env.current_step)
        if done:
            print(info, env.current_step)
            break
    plt.close()
    print(time.time() - start)

if __name__ == '__main__':
    test()