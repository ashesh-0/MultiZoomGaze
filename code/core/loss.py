import numpy as np
import torch
import torch.nn as nn


class PinBallLossBase(nn.Module):
    def __init__(self):
        super(PinBallLossBase, self).__init__()
        self.q1 = 0.1
        self.q9 = 1 - self.q1

    def forward(self, output_o, target_o, var_o):
        q_10 = target_o - (output_o - var_o)
        q_90 = target_o - (output_o + var_o)

        loss_10 = torch.max(self.q1 * q_10, (self.q1 - 1) * q_10)
        loss_90 = torch.max(self.q9 * q_90, (self.q9 - 1) * q_90)
        return loss_10, loss_90


class PinBallLoss(PinBallLossBase):
    def forward(self, output_o, target_o, var_o):
        loss_10, loss_90 = super().forward(output_o, target_o, var_o)
        loss_10 = torch.mean(loss_10)
        loss_90 = torch.mean(loss_90)

        return loss_10 + loss_90


class YawPitchBalancedPinBallLoss(PinBallLossBase):
    def __init__(self, yaw_dim_list, pitch_dim_list):
        super().__init__()
        self._yaw_dim_list = yaw_dim_list
        self._pitch_dim_list = pitch_dim_list
        assert len(set(self._yaw_dim_list).intersection(set(self._pitch_dim_list))) == 0
        assert len(set(self._yaw_dim_list)) == len(self._yaw_dim_list)
        assert len(set(self._pitch_dim_list)) == len(self._pitch_dim_list)
        assert min(min(self._yaw_dim_list), min(self._pitch_dim_list)) == 0
        assert max(max(self._yaw_dim_list),
                   max(self._pitch_dim_list)) == len(self._yaw_dim_list) + len(self._pitch_dim_list) - 1

    def forward(self, output_o, target_o, var_o):
        loss_10, loss_90 = super().forward(output_o, target_o, var_o)
        loss_10 = 1 / 2 * (torch.mean(loss_10[:, self._yaw_dim_list]) + torch.mean(loss_10[:, self._pitch_dim_list]))
        loss_90 = 1 / 2 * (torch.mean(loss_90[:, self._yaw_dim_list]) + torch.mean(loss_90[:, self._pitch_dim_list]))

        return loss_10 + loss_90
