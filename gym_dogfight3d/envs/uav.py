import random
import numpy as np
import math
import transforms3d as trans
from gym_dogfight3d.envs.utils import *

UP = np.array([0, 1, 0], dtype=np.float32)
FRONT = np.array([0, 0, 1], dtype=np.float32)

class UAV:
    """docstring for uav"""

    def __init__(self):
        self.health_level = 1
        self.wreck = False
        self.fallInWater = False
        self.position = np.array([0, 0, 0], dtype=np.float32)
        self.v_move = np.array([0, 0, 0], dtype=np.float32)
        self.thrust_level = 0
        self.thrust_force = 10
        self.angular_frictions = np.array([0.000175, 0.000125, 0.000275], dtype=np.float32)  # pitch, yaw, roll
        self.angular_levels = np.array([0, 0, 0], dtype=np.float32)  # 0 to 1
        self.angular_levels_dest = np.array([0, 0, 0], dtype=np.float32)
        self.angular_levels_inertias = np.array([3, 3, 3], dtype=np.float32)
        self.angular_speed = np.array([0, 0, 0], dtype=np.float32)
        self.speed_ceiling = 1750

        self.wings_max_angle = 45
        self.wings_level = 0
        self.wings_thresholds = np.array([500, 750], dtype=np.float32)
        self.wings_geometry_gain_friction = -0.0001

        self.drag_coeff = np.array([0.033, 0.06666, 0.0002], dtype=np.float32)
        self.wings_lift = 0.0005
        self.brake_level = 0
        self.brake_drag = 0.006
        self.flaps_level = 0
        self.flaps_lift = 0.0025
        self.flaps_drag = 0.002
        self.flag_easy_steering = True
        self.F_gravity = np.array([0, -9.8, 0], dtype=np.float32)
        self.air_density = 1.225

        # Linear acceleration:
        self.linear_acceleration = 9.8
        self.linear_speed = 0

        # Attitudes calculation:
        self.horizontal_aX = None
        self.horizontal_aY = None
        self.horizontal_aZ = None
        self.rotation = np.array([0, 0, 0], dtype=np.float32)
        self.cap = 0
        self.pitch_attitude = 0  # x
        self.roll_attitude = 0  # z
        # self.cap = math.radians(random.uniform(-180, 180))  # y
        self.y_dir = 1
        self.direction = np.array([0, 0, 0], dtype=np.float32)

        # attack
        self.attack_range = [10, 800]
        self.attack_angle = 10

    def reset(self, thrust_level, linear_speed, position=None, rotation=None):
        if position is not None:
            self.position = position
        if rotation is not None:
            self.rotation = rotation
            self.get_attitude(rotation)
        self.set_linear_speed(linear_speed)
        self.set_thrust_level(thrust_level)
        self.wreck = False
        self.fallInWater = False
        self.health_level = 1
        self.angular_levels = np.array([0, 0, 0], dtype=np.float32)  # 0 to 1
        self.angular_levels_dest = np.array([0, 0, 0], dtype=np.float32)

    def get_attitude(self, rotation):
        mat = trans.euler.euler2mat(rotation[0], rotation[1], rotation[2], 'ryxz')
        aX, aY, aZ = mat[:, 0], mat[:, 1], mat[:, 2]
        if aY[1] > 0:
            self.y_dir = 1
        else:
            self.y_dir = -1
        self.horizontal_aZ = normalize(np.array([aZ[0], 0, aZ[2]], dtype=np.float32))
        self.horizontal_aX = np.cross(UP, self.horizontal_aZ) * self.y_dir
        self.horizontal_aY = np.cross(aZ, self.horizontal_aX)

        self.pitch_attitude = math.degrees(math.acos(max(-1, min(1, np.dot(self.horizontal_aZ, aZ)))))
        if aZ[1] < 0: self.pitch_attitude *= -1

        self.roll_attitude = math.degrees(math.acos(max(-1, min(1, np.dot(self.horizontal_aX, aX)))))
        if aX[1] < 0: self.roll_attitude *= -1

        self.cap = math.degrees(math.acos(max(-1, min(1, np.dot(self.horizontal_aZ, FRONT)))))
        if self.horizontal_aZ[0] < 0: self.cap = 360 - self.cap

    def set_linear_speed(self, value):
        aZ = trans.euler.euler2mat(self.rotation[0], self.rotation[1], self.rotation[2], 'ryxz')[:, 2]
        self.v_move = aZ * value
        self.linear_speed = np.linalg.norm(self.v_move)

    def set_thrust_level(self, value):
        self.thrust_level = min(max(value, 0), 1)

    def set_pitch_level(self, value):
        self.angular_levels_dest[0] = max(min(1, value), -1)

    def set_yaw_level(self, value):
        self.angular_levels_dest[1] = max(min(1, value), -1)

    def set_roll_level(self, value):
        self.angular_levels_dest[2] = max(min(1, value), -1)

    def set_wings_level(self, value):
        self.wings_level = min(max(value, 0), 1)

    def stabilize(self, p, y, r):
        if p: self.set_pitch_level(0)
        if y: self.set_yaw_level(0)
        if r: self.set_roll_level(0)

    def hit(self, value):
        if not self.wreck:
            self.health_level = min(max(self.health_level - value, 0), 1)
            if self.health_level == 0 and not self.wreck:
                self.wreck = True

    def update_inertial_value(self, v0, vd, vi, dt):
        vt = vd - v0
        if vt < 0:
            v = v0 - vi * dt
            if v < vd: v = vd
        elif vt > 0:
            v = v0 + vi * dt
            if v > vd: v = vd
        else:
            v = vd
        return v

    def update_angular_levels(self, dt):
        self.angular_levels[0] = self.update_inertial_value(self.angular_levels[0], self.angular_levels_dest[0],
                                                            self.angular_levels_inertias[0], dt)
        self.angular_levels[1] = self.update_inertial_value(self.angular_levels[1], self.angular_levels_dest[1],
                                                            self.angular_levels_inertias[1], dt)
        self.angular_levels[2] = self.update_inertial_value(self.angular_levels[2], self.angular_levels_dest[2],
                                                            self.angular_levels_inertias[2], dt)

    def update_kinetics(self, dt):
        self.update_angular_levels(dt)

        # euler to rotation matrix
        mat = trans.euler.euler2mat(self.rotation[0], self.rotation[1], self.rotation[2], 'ryxz')
        aX, aY, aZ = mat[:, 0], mat[:, 1], mat[:, 2]

        if aY[1] > 0:
            self.y_dir = 1
        else:
            self.y_dir = -1
        self.horizontal_aZ = normalize(np.array([aZ[0], 0, aZ[2]], dtype=np.float32))
        self.horizontal_aX = np.cross(UP, self.horizontal_aZ) * self.y_dir
        self.horizontal_aY = np.cross(aZ, self.horizontal_aX)  # ! It's not an orthogonal repere !

        # axis speed:
        spdX = aX * np.dot(aX, self.v_move)
        spdY = aY * np.dot(aY, self.v_move)
        spdZ = aZ * np.dot(aZ, self.v_move)

        frontal_speed = np.linalg.norm(spdZ)

        # wings_geometry:
        self.set_wings_level(max(min(
            (frontal_speed * 3.6 - self.wings_thresholds[0]) / (self.wings_thresholds[1] - self.wings_thresholds[0]),
            1), 0))

        # Thrust force:
        k = pow(self.thrust_level, 2) * self.thrust_force
        F_thrust = aZ * k

        # Dynamic pressure:
        q = np.array([pow(np.linalg.norm(spdX), 2), pow(np.linalg.norm(spdY), 2),
                      pow(np.linalg.norm(spdZ), 2)], dtype=np.float32) * 0.5 * self.air_density

        # F Lift
        F_lift = aY * q[2] * (self.wings_lift + self.flaps_level * self.flaps_lift)

        # Drag force:
        F_drag = normalize(spdX) * q[0] * self.drag_coeff[0] + normalize(spdY) * q[1] * self.drag_coeff[1] + normalize(
            spdZ) * q[2] * (
                         self.drag_coeff[
                             2] + self.brake_drag * self.brake_level + self.flaps_level * self.flaps_drag + self.wings_geometry_gain_friction * self.wings_level)

        # Total
        self.v_move += ((F_thrust + F_lift - F_drag) * self.health_level + self.F_gravity) * dt

        # acceleration&speed
        self.linear_acceleration = (np.linalg.norm(self.v_move) - self.linear_speed) / dt
        self.linear_speed = np.linalg.norm(self.v_move)

        # Displacement:
        self.position += self.v_move * dt

        # Rotations:
        F_pitch = self.angular_levels[0] * q[2] * self.angular_frictions[0]
        F_yaw = self.angular_levels[1] * q[2] * self.angular_frictions[1]
        F_roll = self.angular_levels[2] * q[2] * self.angular_frictions[2]

        # Angular damping:
        gaussian = math.exp(-pow(frontal_speed * 3.6 * 3 / self.speed_ceiling, 2) / 2)

        # Angular speed:
        self.angular_speed = np.array([F_pitch, F_yaw, F_roll], dtype=np.float32) * gaussian

        # Moment:
        pitch_m = aX * self.angular_speed[0]
        yaw_m = aY * self.angular_speed[1]
        roll_m = aZ * self.angular_speed[2]

        # Easy steering:
        if self.flag_easy_steering:

            easy_yaw_angle = (1 - (np.dot(aX, self.horizontal_aX)))
            if np.dot(aZ, np.cross(aX, self.horizontal_aX)) < 0:
                easy_turn_m_yaw = self.horizontal_aY * -easy_yaw_angle
            else:
                easy_turn_m_yaw = self.horizontal_aY * easy_yaw_angle

            easy_roll_stab = np.cross(aY, self.horizontal_aY) * self.y_dir
            if self.y_dir < 0:
                easy_roll_stab = normalize(easy_roll_stab)
            else:
                n = np.linalg.norm(easy_roll_stab)
                if n > 0.1:
                    easy_roll_stab = normalize(easy_roll_stab)
                    easy_roll_stab *= (1 - n) * n + n * pow(n, 0.125)

            zl = min(1, abs(self.angular_levels[2] + self.angular_levels[0] + self.angular_levels[1]))
            roll_m += (easy_roll_stab * (1 - zl) + easy_turn_m_yaw) * q[2] * self.angular_frictions[1] * gaussian

        # Moment:
        moment = yaw_m + roll_m + pitch_m
        axis_rot = normalize(moment)
        moment_speed = np.linalg.norm(moment) * self.health_level

        # Rotation matrix:
        rot_mat = rotate_matrix(mat, axis_rot, moment_speed * dt)
        mat = np.transpose(rot_mat)
        self.direction = mat[:, 2]

        # pitch, row, cap
        self.rotation = trans.euler.mat2euler(mat, 'ryxz')
        self.get_attitude(self.rotation)

        # fall into water:
        if self.position[1] < 4:
            self.fallInWater = True
            self.hit(1)


if __name__ == '__main__':
    uav = UAV()
    axes = {
        'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
        'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
        'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
        'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
        'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
        'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
        'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
        'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)
    }
    ret = trans.euler.euler2mat(3, -0.2, 0.5, 'ryxz')
    print(ret[:,0],
          ret[:,1],
          ret[:,2], sep='\n')
    for axe in axes:
        ret2 = trans.euler.euler2mat(3, -0.2, 0.5, axe)
        if abs(ret2[0, 0] - ret[0, 0]) < 0.0001:
            print(axe)
            print(ret[:,0],
                  ret[:,1],
                  ret[:,2], sep='\n')
    # thrust_level = 0.7
    # linear_speed = 800 / 3.6
    # postion = np.array([0., 3000., 0.])
    # rotation = np.array([1, 0, 0.5])
    # uav.reset(thrust_level, linear_speed, postion, rotation)
    # for i in range(1000):
    #
    #     uav.update_kinetics(0.08)
