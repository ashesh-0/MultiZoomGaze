import torch
import torch.nn as nn

from core.loss import PinBallLoss, PinBallLossBase, YawPitchBalancedPinBallLoss


def weighted_loss(batch_loss, target, w_lambda):
    w_for_sinYaw = torch.abs(target[:, 0]) + w_lambda
    w_for_cosYaw = torch.abs(target[:, 1]) + w_lambda
    w_for_sinPitch = torch.abs(target[:, 2]) + w_lambda
    batch_loss[:, 0] = batch_loss[:, 0] * w_for_sinYaw
    batch_loss[:, 1] = batch_loss[:, 1] * w_for_cosYaw
    batch_loss[:, 2] = batch_loss[:, 2] * w_for_sinPitch
    return batch_loss


class WeightedMseLoss(nn.Module):
    def __init__(self, w_lambda):
        super().__init__()
        self.w_lambda = w_lambda

    def forward(self, output_o, target_o, var_o):
        loss = (output_o - target_o)**2
        loss = weighted_loss(loss, target_o, self.w_lambda)
        return torch.mean(loss)


class WeightedPinBallLoss(PinBallLossBase):
    """
    sin(90) - sin(75) = 0.03
    sin(15) - sin(0) = 0.25
    In our problem, we are dealing with sin(angle) but we want to optimize angle. So we need to weight the samples.
    weight for sin(Yaw) is sin(Yaw)
    weight for cos(Yaw) is cos(Yaw)
    weight for sin(Pitch) is sin(Pitch)
    """

    def __init__(self, w_lambda):
        super(WeightedPinBallLoss, self).__init__()
        self.w_lambda = w_lambda

    def forward(self, output_o, target_o, var_o):
        loss_10, loss_90 = super().forward(output_o, target_o, var_o)

        loss_10 = weighted_loss(loss_10, target_o, self.w_lambda)
        loss_90 = weighted_loss(loss_90, target_o, self.w_lambda)
        loss_10 = torch.mean(loss_10)
        loss_90 = torch.mean(loss_90)

        return loss_10 + loss_90


class SinCosLoss(nn.Module):
    """
    Pinball loss on first two columns which are sin(pitch) and sin/cos(yaw)
    CrossEntropy loss on boolean signal.
    """

    def __init__(self):
        super(SinCosLoss, self).__init__()
        self.pinball_loss = PinBallLoss()
        self.crossentropy_loss = nn.BCELoss()
        self._pinball_weight = 0.9

    def forward(self, output_o, target_o, var_o):
        # pinball loss.
        pb_loss = self.pinball_loss.forward(output_o[:, :2], target_o[:, :2], var_o)
        # binary loss.
        ce_loss = self.crossentropy_loss(output_o[:, 2:3], target_o[:, 2:3])

        return self._pinball_weight * pb_loss + (1 - self._pinball_weight) * ce_loss


class AngularLoss(nn.Module):
    def forward(self, output_o, target_o):
        # without normalization loss ensuring vector has norm 1, we have a difficulty here.

        sincosYaw_norm = torch.norm(output_o[:, :2], dim=1).view(-1, 1)
        sincosYaw = output_o[:, :2] / sincosYaw_norm
        # import pdb
        # pdb.set_trace()
        # sin(Yaw),cos(Yaw),sin(Pitch)
        o_cosP = torch.sqrt(1 - output_o[:, 2]**2)
        t_cosP = torch.sqrt(1 - target_o[:, 2]**2)

        o_cosPsinY = o_cosP * sincosYaw[:, 0]
        t_cosPsinY = t_cosP * target_o[:, 0]

        o_cosPcosY = o_cosP * sincosYaw[:, 1]
        t_cosPcosY = t_cosP * target_o[:, 1]

        cosangle = output_o[:, 2] * target_o[:, 2] + o_cosPcosY * t_cosPcosY + o_cosPsinY * t_cosPsinY

        sinangle = 1 - cosangle**2
        return torch.mean(sinangle)


class RegularizedSinAndCosLoss(nn.Module):
    """
    Pinball loss on all columns which are sin/cos(pitch) and sin/cos(yaw)
    MSE loss enforcing sin2(yaw) + cos2(yaw) =1. sin(yaw) and cos(yaw) are first two columns.
    """

    def __init__(self):
        super(RegularizedSinAndCosLoss, self).__init__()
        self.pinball_loss = PinBallLoss()
        self.mse_loss = nn.MSELoss()
        self._pinball_weight = 0.9

    def forward(self, output_o, target_o, var_o):
        # pinball loss.
        pb_loss = self.pinball_loss.forward(output_o, target_o, var_o)
        # `sin2(yaw) + cos2(yaw) must be 1` loss.
        pred_norm = torch.norm(output_o[:, :2], dim=1)
        mse_loss = self.mse_loss(pred_norm, torch.ones(*pred_norm.shape).cuda())

        return self._pinball_weight * pb_loss + (1 - self._pinball_weight) * mse_loss


class RegularizedMultiSinAndCosLoss(RegularizedSinAndCosLoss):
    def __init__(self, yaw_dim_list, pitch_dim_list):
        super().__init__()
        self.pinball_loss = YawPitchBalancedPinBallLoss(yaw_dim_list, pitch_dim_list)


class RegularizedSinAndCosLossWithAngularLoss(RegularizedSinAndCosLoss):
    def __init__(self, angular_weight):
        super().__init__()
        self._angular_loss = AngularLoss()
        self._angular_W = angular_weight
        print(f'[{self.__class__.__name__}] W:{self._angular_W}')

    def forward(self, output_o, target_o, var_o):
        pb_loss = super().forward(output_o, target_o, var_o)
        ang_loss = self._angular_loss(output_o, target_o)
        # pred_norm = torch.norm(output_o[:, :2], dim=1)
        # mse_loss = self.mse_loss(pred_norm, torch.ones(*pred_norm.shape).cuda())
        # import pdb
        # pdb.set_trace()

        return (1 - self._angular_W) * pb_loss + self._angular_W * (ang_loss)


class WeightedRegularizedSinAndCosLoss(RegularizedSinAndCosLoss):
    """
    Pinball loss on all columns which are sin/cos(pitch) and sin/cos(yaw)
    MSE loss enforcing sin2(yaw) + cos2(yaw) =1. sin(yaw) and cos(yaw) are first two columns.
    """

    def __init__(self, w_lambda=0.01):
        super(WeightedRegularizedSinAndCosLoss, self).__init__()
        self.pinball_loss = WeightedPinBallLoss(w_lambda)


if __name__ == '__main__':
    import numpy as np
    loss = AngularLoss()
    target = torch.Tensor([
        [1 / np.sqrt(2), 1 / np.sqrt(2), 0],  #45,0
        [1 / np.sqrt(2), 1 / np.sqrt(2), np.sqrt(3) / 2],  #45,60
        [1 / np.sqrt(2), 1 / np.sqrt(2), -1 / 2],  #45,-30
        [1 / 2, np.sqrt(3) / 2, 1 / 2],  #30,30
    ])

    predicted = torch.Tensor([
        [1 / np.sqrt(2), 1 / np.sqrt(2), 0],  #45,0
        [1 / np.sqrt(2), 1 / np.sqrt(2), np.sqrt(3) / 2],  #45,60
        [1 / np.sqrt(2), 1 / np.sqrt(2), -1 / 2],  #45,-30
        [1 / 2, np.sqrt(3) / 2, 1 / 2],  #30,30
    ])

    print(loss(target, predicted))
    print(loss(predicted, target))

    print('')
    predicted = torch.Tensor([
        [1, 0, 0],  #90,0
        [0, 1, np.sqrt(3) / 2],  #0,60
        [1 / np.sqrt(2), 1 / np.sqrt(2), -1 / 2],  #45,-30
        [1 / 2, np.sqrt(3) / 2, 1 / 2],  #30,30
    ])

    print(loss(target, predicted))
    print(loss(predicted, target))
