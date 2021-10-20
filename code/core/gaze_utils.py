import math
from typing import List

import numpy as np
import torch
import torch.nn as nn


def spherical2cartesial(x: np.array) -> np.array:
    """
    Args:
        x: N*2 containing the yaw and pitch angles.
    Returns:
        vector in xyz coordinates.
    """
    output = np.zeros((x.shape[0], 3))
    cos_0 = np.apply_along_axis(math.cos, 1, x[:, :1])
    cos_1 = np.apply_along_axis(math.cos, 1, x[:, 1:])

    sin_0 = np.apply_along_axis(math.sin, 1, x[:, :1])
    sin_1 = np.apply_along_axis(math.sin, 1, x[:, 1:])

    output[:, 2] = -cos_1 * cos_0
    output[:, 0] = cos_1 * sin_0
    output[:, 1] = sin_1

    return output


def compute_angular_error_arr(input_: np.array, target_: np.array) -> float:
    input_ = spherical2cartesial(input_)
    target_ = spherical2cartesial(target_)

    input_ = input_.reshape((-1, 3, 1))
    target_ = target_.reshape((-1, 1, 3))
    output_dot = np.matmul(target_, input_)
    output_dot = output_dot.reshape((-1, 1))
    output_dot = 180 * np.apply_along_axis(math.acos, 1, output_dot) / math.pi
    return output_dot


def compute_angular_error(input_: np.array, target_: np.array) -> float:
    """
    Args:
        input_: N*2 numpy array containing the predicted yaw and pitch angle
        target_: N*2 numpy array containing the corresponding target_ yaw and pitch angle.
    Returns:
        Average angular error (in degree)
    """

    return np.mean(compute_angular_error_arr(input_, target_))


def get_spherical_vector(xyz_gaze: List[float]) -> List[float]:
    """
    Args:
        xyz_gaze: Array of size 3 containing normalized vectors. Note that normalization must have been done.
    Returns:
        Returns array of size 2 containing yaw and pitch
    """
    sph = [None, None]
    # yaw
    sph[0] = math.atan2(xyz_gaze[0], -xyz_gaze[2])
    # pitch
    sph[1] = math.asin(xyz_gaze[1])
    return sph


def get_yaw_pitch(gaze: np.array) -> torch.Tensor:
    gaze_float = torch.Tensor(gaze)
    gaze_float = torch.FloatTensor(gaze_float)
    normalized_gaze = nn.functional.normalize(gaze_float.view(1, 3)).view(3)

    spherical_vector = torch.FloatTensor(2)
    # yaw
    spherical_vector[0] = math.atan2(normalized_gaze[0], -normalized_gaze[2])
    # pitch
    spherical_vector[1] = math.asin(normalized_gaze[1])
    return spherical_vector


def average_angle(angle1: torch.Tensor, angle2: torch.Tensor) -> torch.Tensor:
    """
    Args:
        angle1: one dimensional tensor in radians
        angle2: one dimensional tensor in radians
    """
    c1 = torch.abs(angle1 - 2 * math.pi - angle2).view(-1, 1)
    c2 = torch.abs(angle1 - angle2).view(-1, 1)
    c3 = torch.abs(angle1 + 2 * math.pi - angle2).view(-1, 1)
    c123 = torch.cat([c1, c2, c3], dim=1)
    factor = torch.argmin(c123, dim=1) - 1

    angle1 = angle1 + factor * 2 * math.pi

    avg_angle = (angle1 + angle2) / 2

    avg_angle[avg_angle > math.pi] = avg_angle[avg_angle > math.pi] - 2 * math.pi
    avg_angle[avg_angle < -math.pi] = avg_angle[avg_angle < -math.pi] + 2 * math.pi
    return avg_angle
