import random
import numpy as np
import math
import transforms3d as trans


def normalize(v):
    l2 = np.linalg.norm(v)
    if l2 == 0: return np.array([0, 0, 0], dtype=np.float32)
    return v / l2

def rotate_vector(point, axe, angle):
    normalize(axe)
    dot_prod = point[0] * axe[0] + point[1] * axe[1] + point[2] * axe[2]
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)

    return np.array(
        [cos_angle * point[0] + sin_angle * (axe[1] * point[2] - axe[2] * point[1]) + (1 - cos_angle) * dot_prod * axe[0], \
        cos_angle * point[1] + sin_angle * (axe[2] * point[0] - axe[0] * point[2]) + (1 - cos_angle) * dot_prod * axe[1], \
        cos_angle * point[2] + sin_angle * (axe[0] * point[1] - axe[1] * point[0]) + (1 - cos_angle) * dot_prod * axe[2]], dtype=np.float32)

def rotate_matrix(mat, axe, angle):
    axeX, axeY, axeZ = mat[:, 0], mat[:, 1], mat[:, 2]
    axeXr = rotate_vector(axeX, axe, angle)
    axeYr = rotate_vector(axeY, axe, angle)
    axeZr = np.cross(axeXr, axeYr)  # cls.rotate_vector(axeZ,axe,angle)
    return np.array([axeXr, axeYr, axeZr], dtype=np.float32)