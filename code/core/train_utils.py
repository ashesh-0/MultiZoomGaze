import math
import os
import shutil
from typing import Tuple

import torch
import torch.nn as nn

epsilon = 1e-6


def inv_tanh(inp):
    return torch.log((1 + inp) / (1 - inp)) / 2


def save_checkpoint(state, is_best, fpath):
    torch.save(state, fpath)
    if is_best:
        dirname = os.path.dirname(fpath)
        fname = os.path.basename(fpath)
        best_fpath = os.path.join(dirname, 'model_best_' + fname)
        shutil.copyfile(fpath, best_fpath)


def cartesialtospherical(gaze_xyz):
    gaze_float = torch.Tensor(gaze_xyz)
    gaze_float = torch.FloatTensor(gaze_float)
    normalized_gaze = nn.functional.normalize(gaze_float)

    spherical_vector = torch.FloatTensor(normalized_gaze.size()[0], 2)
    # yaw
    spherical_vector[:, 0] = torch.atan2(normalized_gaze[:, 0], -normalized_gaze[:, 2])
    # pitch
    spherical_vector[:, 1] = torch.asin(normalized_gaze[:, 1])
    return spherical_vector


def spherical2cartesial(input):
    z = -torch.cos(input[:, 1]) * torch.cos(input[:, 0])
    x = torch.cos(input[:, 1]) * torch.sin(input[:, 0])
    y = torch.sin(input[:, 1])

    return torch.cat((x[:, None], y[:, None], z[:, None]), dim=1)


def compute_angular_error_xyz(input, target):
    return torch.mean(compute_angular_error_xyz_arr(input, target))


def compute_angular_error_xyz_arr(input, target):
    input = input.view(-1, 3, 1)
    input = input / (torch.norm(input[:, :, 0], p=2, dim=1).view(-1, 1, 1) + epsilon)

    target = target.view(-1, 1, 3)
    output_dot = torch.bmm(target, input)
    output_dot = output_dot.view(-1)
    output_dot = torch.acos(output_dot)
    output_dot = 180 * output_dot / math.pi
    return output_dot


def compute_angular_error(input, target):
    return torch.mean(compute_angular_error_arr(input, target))


def compute_angular_error_arr(input, target):
    input = spherical2cartesial(input)
    target = spherical2cartesial(target)
    return compute_angular_error_xyz_arr(input, target)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def checkpoint_fname_noext(static: bool, kfold, keys_list: Tuple[str, object], equalto_str=':'):
    fname = 'multiZoomGaze'
    if static:
        fname += '_static'
    if kfold is not None and kfold >= 0:
        fname += f'_Kfold{equalto_str}{kfold}'
    for key, val in keys_list:
        fname += f'_{key}{equalto_str}{val}'
    return fname


def checkpoint_fname(static: bool, kfold, keys_list: Tuple[str, object], dirname='/home/ashesh/'):
    return os.path.join(dirname, checkpoint_fname_noext(static, kfold, keys_list) + '.pth.tar')


def checkpoint_params(fname):
    fname = os.path.basename(fname)

    static = '_static_' in fname
    assert fname[-8:] == '.pth.tar'
    fname = fname[:-8]
    param_prefix = 'multiZoomGaze'
    if static:
        param_prefix += '_static'
    param_prefix += '_'

    idx = fname.find(param_prefix)
    fname = fname[idx + len(param_prefix):]
    tokens = fname.split(':')
    keys = [tokens[0]]
    values = []
    for token in tokens[1:]:
        values.append(token.split('_')[0])
        keys.append('_'.join(token.split('_')[1:]))

    return {k: v for k, v in zip(keys, values)}


if __name__ == '__main__':
    sin60 = math.sqrt(3) / 2
    sin30 = 1 / 2
    sin45 = 1 / math.sqrt(2)

    a = torch.Tensor([[45 * math.pi / 180, 0]])
    b = torch.Tensor([[120 * math.pi / 180, 0]])
    print('Error', compute_angular_error(a, b))
