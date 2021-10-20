import math

import numpy as np
import torch

from core.gaze_utils import average_angle, get_yaw_pitch


def test_get_yaw_pitch():
    assert np.linalg.norm(np.array([math.pi / 2, 0]) - get_yaw_pitch(np.array([1, 0, 0])).numpy()) < 1e-5
    assert np.linalg.norm(np.array([-math.pi / 2, 0]) - get_yaw_pitch(np.array([-1, 0, 0])).numpy()) < 1e-5
    assert np.linalg.norm(
        np.array([0, math.pi / 4]) - get_yaw_pitch(np.array([0, 1 / math.sqrt(2), -1 / math.sqrt(2)])).numpy()) < 1e-5
    assert np.linalg.norm(np.array([math.pi, 0]) - get_yaw_pitch(np.array([0, 0, 1])).numpy()) < 1e-5


def test_average_angle():
    p = math.pi
    a = torch.Tensor([0, p / 4, p / 2, p - 0.01, p - 0.03])
    b = torch.Tensor([-p / 4, -p / 4, -3 * p / 4, -p + 0.03, -p + 0.01])

    avg = average_angle(a, b)
    expected = torch.Tensor([-p / 8, 0, 7 * p / 8, -p + 0.01, p - 0.01])
    assert torch.max(torch.abs(avg - expected)).item() < 1e-6
