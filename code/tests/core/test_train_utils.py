import math

import numpy as np
import torch
from mock import patch

from core.train_utils import compute_angular_error, compute_angular_error_xyz_arr, spherical2cartesial


def test_spherical2cartesial():
    spherical = torch.Tensor([
        [0, 0],
        [math.pi / 2, 0],
        [-math.pi / 2, 0],
        [0, math.pi / 2],
        [math.pi / 2, math.pi / 2],
    ])
    target_xyz = np.array([
        [0, 0, -1],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
    ])
    xyz = spherical2cartesial(spherical)
    assert xyz.shape[0] == spherical.shape[0]
    assert xyz.shape[1] == 3
    assert isinstance(xyz, torch.Tensor)
    assert np.linalg.norm(target_xyz - xyz.numpy(), axis=1).max() < 1e-5


@patch('core.train_utils.epsilon', 0)
def test_compute_angular_error_xyz_arr():
    input1 = torch.Tensor([
        [0.8001 / math.sqrt(2), 0.6, 0.8 / math.sqrt(2)],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [-1, 0, 0],
        [1 / math.sqrt(2), 0, 1 / math.sqrt(2)],
    ])
    input2 = torch.Tensor([
        [0.8 / math.sqrt(2), 0.6, 0.8 / math.sqrt(2)],
        [-1, 0, 0],
        [0, 0, 1],
        [0, 0, -1],
        [-1, 0, 0],
        [-1 / math.sqrt(2), 0, 1 / math.sqrt(2)],
    ], )
    target = torch.Tensor([
        0,
        180,
        180 / 2,
        180,
        0,
        180 / 2,
    ])
    output = compute_angular_error_xyz_arr(input1, input2)
    assert np.max(np.abs(output.numpy() - target.numpy())) < 1e-5


@patch('core.train_utils.epsilon', 0)
def test_compute_angular_error():
    input1 = torch.Tensor([
        [math.pi / 2, 0],
        [0, math.pi / 2],
        [math.pi, 0],
        [-math.pi / 2, 0],
        [math.pi / 4, 0],
    ])
    input2 = torch.Tensor([
        [-math.pi / 2, 0],
        [math.pi, 0],
        [0, 0],
        [-math.pi / 2, 0],
        [-math.pi / 4, 0],
    ], )
    target = torch.Tensor([
        180,
        180 / 2,
        180,
        0,
        180 / 2,
    ])
    output = compute_angular_error(input1, input2)
    assert torch.mean(target) == output
