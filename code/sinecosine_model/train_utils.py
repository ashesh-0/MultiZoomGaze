import math

import torch

from core.gaze_utils import average_angle
from core.train_utils import compute_angular_error_arr


def get_multi_sincos_gaze_from_spherical(spherical_vector: torch.Tensor):
    gaze_sinecos_vector = torch.Tensor([0., 0., 0., 0., 0.])
    #sin(Yaw)
    gaze_sinecos_vector[0] = torch.sin(spherical_vector[0])
    # cos(Yaw)
    gaze_sinecos_vector[1] = torch.cos(spherical_vector[0])
    # sin(Pitch)
    gaze_sinecos_vector[2] = torch.sin(spherical_vector[1])

    # sin(Yaw + 45)
    gaze_sinecos_vector[3] = torch.sin(spherical_vector[0] + math.pi / 4)
    # cos(Yaw + 45)
    gaze_sinecos_vector[4] = torch.cos(spherical_vector[0] + math.pi / 4)

    return gaze_sinecos_vector


def get_sincos_gaze_from_spherical(spherical_vector: torch.Tensor):
    gaze_sinecos_vector = torch.Tensor([0., 0., 0.])
    #sin(Yaw)
    gaze_sinecos_vector[0] = torch.sin(spherical_vector[0])
    # cos(Yaw)
    gaze_sinecos_vector[1] = torch.cos(spherical_vector[0])
    # sin(Pitch)
    gaze_sinecos_vector[2] = torch.sin(spherical_vector[1])
    return gaze_sinecos_vector


def get_cosinebased_yaw_pitch(input_: torch.Tensor) -> torch.Tensor:
    """
    Returns a tensor with two columns being yaw and pitch respectively. For yaw, it uses cos(yaw)'s value along with
    sin(yaw)'s sign.
    Args:
        input_: 1st column is sin(yaw), 2nd Column is cos(yaw), 3rd Column is sin(pitch)
    """

    yaw_pitch_cosine = torch.zeros((input_.shape[0], 2))
    yaw_pitch_cosine[:, 1] = torch.asin(input_[:, 2])

    yaw = torch.acos(input_[:, 1])
    right = (input_[:, 0] < 0.)
    yaw[right] = -1 * yaw[right]

    yaw_pitch_cosine[:, 0] = yaw
    return yaw_pitch_cosine


def get_sinebased_yaw_pitch(input_: torch.Tensor) -> torch.Tensor:
    """
    Returns a tensor with two columns being yaw and pitch respectively. For yaw, it uses sin(yaw)'s value along with
    cos(yaw)'s sign.
    Args:
        input_: 1st column is sin(yaw), 2nd Column is cos(yaw), 3rd Column is sin(pitch)
    """

    yaw_pitch_sine = torch.zeros((input_.shape[0], 2))
    yaw_pitch_sine[:, 1] = torch.asin(input_[:, 2])

    sin_based_yaw = torch.asin(input_[:, 0])
    back = (input_[:, 1] < 0)
    pos_yaw = sin_based_yaw >= 0

    pos_back = pos_yaw & back
    neg_back = (~pos_yaw) & back

    sin_based_yaw[pos_back] = math.pi - sin_based_yaw[pos_back]
    sin_based_yaw[neg_back] = -math.pi - sin_based_yaw[neg_back]

    yaw_pitch_sine[:, 0] = sin_based_yaw
    return yaw_pitch_sine


def compute_angular_error_sine_and_cosine(input_: torch.Tensor, target_: torch.Tensor) -> torch.Tensor:
    return torch.mean(compute_angular_error_sine_and_cosine_arr(input_, target_))


def compute_yaw_pitch(input_: torch.Tensor):
    yaw_pitch_cosine = get_cosinebased_yaw_pitch(input_)
    yaw_pitch_sine = get_sinebased_yaw_pitch(input_)
    pred_yaw = average_angle(yaw_pitch_cosine[:, 0], yaw_pitch_sine[:, 0]).view(-1, 1)
    pred_pitch = average_angle(yaw_pitch_cosine[:, 1], yaw_pitch_sine[:, 1]).view(-1, 1)
    pred = torch.cat([pred_yaw, pred_pitch], dim=1)
    return pred


def compute_angular_error_sine_and_cosine_arr(input_: torch.Tensor, target_: torch.Tensor) -> torch.Tensor:
    """
    Returns a scalar tensor computing the angular error between input_ and target_. For yaw, it takes a simple average
    of yaw predicted from {cos(yaw),sign(sin(yaw))} and {sin(yaw),sign(cos(yaw))}
    Args:
        input_: 1st column is sin(yaw), 2nd Column is cos(yaw), 3rd Column is sin(pitch)
        target_: Same as input_
    """
    yaw_pitch_cosine = get_cosinebased_yaw_pitch(input_)
    yaw_pitch_sine = get_sinebased_yaw_pitch(input_)
    pred_yaw = average_angle(yaw_pitch_cosine[:, 0], yaw_pitch_sine[:, 0]).view(-1, 1)
    pred_pitch = average_angle(yaw_pitch_cosine[:, 1], yaw_pitch_sine[:, 1]).view(-1, 1)

    pred = torch.cat([pred_yaw, pred_pitch], dim=1)
    target = get_cosinebased_yaw_pitch(target_)
    # import pdb
    # pdb.set_trace()
    return compute_angular_error_arr(pred, target)


if __name__ == '__main__':
    sin60 = math.sqrt(3) / 2
    sin30 = 1 / 2
    sin45 = 1 / math.sqrt(2)

    a = torch.Tensor([
        [-sin60, sin30, 0],
        [sin45, sin45, 0],
    ])
    b = torch.Tensor([
        [sin45, sin45, 0],
        [sin60, -sin30, 0],
    ])
    print('a', a)
    print('a: cosine based', get_cosinebased_yaw_pitch(a) * (180 / math.pi))
    print('a: sine based', get_sinebased_yaw_pitch(a) * (180 / math.pi))
    print('b', b)
    print('b: cosine based', get_cosinebased_yaw_pitch(b) * (180 / math.pi))
    print('b: sine based', get_sinebased_yaw_pitch(b) * (180 / math.pi))
    print('Error:', compute_angular_error_sine_and_cosine(a, b))
