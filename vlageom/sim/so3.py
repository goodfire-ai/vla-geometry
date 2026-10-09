"""Rotation helpers (3x3 rotation matrices, robot base frame).
"""
from __future__ import annotations

import numpy as np


def so3_angle(R) -> float:
    """Rotation angle (rad, in [0, pi]) of rotation matrix R: arccos((tr R - 1) / 2)."""
    R = np.asarray(R, np.float64)
    return float(np.arccos(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)))


def so3_axis(R):
    """Unit rotation axis of R from the antisymmetric part (R - R^T); zeros if degenerate.
    Ill-conditioned near angle 0 and pi."""
    R = np.asarray(R, np.float64)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros(3)


def so3_geodesic(Ra, Rb) -> float:
    """Geodesic distance (rad) between two rotations: the angle of Ra^T Rb."""
    return so3_angle(np.asarray(Ra, np.float64).T @ np.asarray(Rb, np.float64))


def signed_angle_about(R0, Rk, uhat) -> float:
    """Angle (rad) of the relative rotation R0^T Rk, signed by the direction of its axis along
    `uhat`; in [-pi, pi]."""
    rel = np.asarray(R0, np.float64).T @ np.asarray(Rk, np.float64)
    ang = so3_angle(rel)
    if ang < 1e-7:
        return 0.0
    s = float(np.dot(so3_axis(rel), np.asarray(uhat, np.float64)))
    return ang * (1.0 if s >= 0 else -1.0)
