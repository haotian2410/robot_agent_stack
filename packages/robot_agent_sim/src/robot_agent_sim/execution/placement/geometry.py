from __future__ import annotations

import math


def aabb(position, dimensions):
    return ((position[0] - dimensions[0] / 2, position[0] + dimensions[0] / 2),
            (position[1] - dimensions[1] / 2, position[1] + dimensions[1] / 2))


def overlap(a, b, gap: float = 0.005) -> bool:
    return not (a[0][1] + gap <= b[0][0] or b[0][1] + gap <= a[0][0]
                or a[1][1] + gap <= b[1][0] or b[1][1] + gap <= a[1][0])


def quaternion_inverse_rotate(point, quaternion):
    """Rotate a world-relative point into a body-local frame."""
    if not quaternion:
        return tuple(float(value) for value in point)
    w, x, y, z = (float(value) for value in quaternion)
    # q^-1 * p * q, expanded for a unit quaternion.
    px, py, pz = (float(value) for value in point)
    qx = -x; qy = -y; qz = -z
    tx = 2.0 * (qy * pz - qz * py)
    ty = 2.0 * (qz * px - qx * pz)
    tz = 2.0 * (qx * py - qy * px)
    return (px + w * tx + (qy * tz - qz * ty),
            py + w * ty + (qz * tx - qx * tz),
            pz + w * tz + (qx * ty - qy * tx))


def quaternion_rotate(point, quaternion):
    if not quaternion:
        return tuple(float(value) for value in point)
    w, x, y, z = (float(value) for value in quaternion)
    px, py, pz = (float(value) for value in point)
    tx = 2.0 * (y * pz - z * py)
    ty = 2.0 * (z * px - x * pz)
    tz = 2.0 * (x * py - y * px)
    return (px + w * tx + (y * tz - z * ty),
            py + w * ty + (z * tx - x * tz),
            pz + w * tz + (x * ty - y * tx))


def add(a, b):
    return tuple(float(x) + float(y) for x, y in zip(a, b))


def distance(a, b):
    return math.dist(tuple(a), tuple(b))
