import torch
import torch.nn as nn


class ExpWarpLoss(nn.Module):
    def __init__(self, yaw_warp, pitch_warp):
        super().__init__()
        assert yaw_warp * pitch_warp == 0
        assert yaw_warp + pitch_warp > 0
        self._Pwarp = pitch_warp > 0
        print(f'[{self.__class__.__name__}] PitchWarp:{self._Pwarp}')

    def forward(self, warpless, lwarp, rwarp, target):
        # For Yaw, lwarp > warpless > rwarp
        # For Pitch, lwarp ~ upwarp, rwarp ~ downwarp.
        warpless = warpless[:, int(self._Pwarp)]
        lwarp = lwarp[:, int(self._Pwarp)]
        rwarp = rwarp[:, int(self._Pwarp)]
        loss_1 = torch.clamp(torch.exp(warpless - lwarp) - 1, min=0)
        loss_2 = torch.clamp(torch.exp(rwarp - warpless) - 1, min=0)
        return (torch.mean(loss_1) + torch.mean(loss_2)) / 2


class ExpWarp2Loss(ExpWarpLoss):
    def forward(self, warpless, lwarp, rwarp, target):
        # For Yaw, lwarp > warpless > rwarp
        warpless_ord = warpless[:, int(self._Pwarp)]
        lwarp_ord = lwarp[:, int(self._Pwarp)]
        rwarp_ord = rwarp[:, int(self._Pwarp)]
        loss_1 = torch.clamp(torch.exp(warpless_ord - lwarp_ord) - 1, min=0)
        loss_2 = torch.clamp(torch.exp(rwarp_ord - warpless_ord) - 1, min=0)

        # consistency loss
        warpless_cons = warpless[:, 1 - self._Pwarp]
        lwarp_cons = lwarp[:, 1 - self._Pwarp]
        rwarp_cons = rwarp[:, 1 - self._Pwarp]
        loss_3 = (torch.abs(warpless_cons - lwarp_cons) + torch.abs(warpless_cons - rwarp_cons)) / 2
        return (torch.mean(loss_1) + torch.mean(loss_2) + torch.mean(loss_3)) / 3


class ExpWarp4Loss(ExpWarpLoss):
    def forward(self, warpless, lwarp, rwarp, target):
        # For Yaw, lwarp > warpless > rwarp
        # For Pitch, lwarp ~ upwarp, rwarp ~ downwarp.
        warpless_ord = warpless[:, int(self._Pwarp)]
        lwarp_ord = lwarp[:, int(self._Pwarp)]
        rwarp_ord = rwarp[:, int(self._Pwarp)]
        loss_1 = torch.clamp(warpless_ord - lwarp_ord, min=0)
        loss_2 = torch.clamp(rwarp_ord - warpless_ord, min=0)

        # consistency loss
        warpless_cons = warpless[:, 1 - self._Pwarp]
        lwarp_cons = lwarp[:, 1 - self._Pwarp]
        rwarp_cons = rwarp[:, 1 - self._Pwarp]
        loss_3 = (torch.abs(warpless_cons - lwarp_cons) + torch.abs(warpless_cons - rwarp_cons)) / 2

        return (torch.mean(loss_1) + torch.mean(loss_2) + torch.mean(loss_3)) / 3


class ExpWarp5Loss(ExpWarp2Loss):
    def __init__(self, yaw_warp, pitch_warp, yaw_range, pitch_range):
        super().__init__(yaw_warp, pitch_warp)
        assert yaw_range is None or all([len(elem) == 2 for elem in yaw_range])
        assert pitch_range is None or all([len(elem) == 2 for elem in pitch_range])
        self._y_range = yaw_range
        self._p_range = pitch_range
        print(f'[{self.__class__.__name__}] Yrange:{self._y_range} Prange:{self._p_range}')

    def in_range(self, for_pitch, target, angle_range):
        filtr = None
        assert for_pitch in [0, 1]
        for low, high in angle_range:
            temp_filtr = (target[:, for_pitch] >= low) * (target[:, for_pitch] <= high)
            if filtr is None:
                filtr = temp_filtr
            else:
                filtr = filtr + temp_filtr
        return filtr > 0

    def relevant_yaw(self, target):
        if self._y_range is None:
            return True
        return self.in_range(0, target, self._y_range)

    def relevant_pitch(self, target):
        if self._p_range is None:
            return True
        return self.in_range(1, target, self._p_range)

    def forward(self, warpless, lwarp, rwarp, target):
        # find relevant entries
        yaw_filtr = self.relevant_yaw(target)
        pitch_filtr = self.relevant_pitch(target)
        filtr = yaw_filtr * pitch_filtr
        if filtr.int().max() == 0:
            return torch.mean(torch.Tensor([0]).to(target.device))
        filtr = filtr.bool()
        warpless = warpless[filtr]
        lwarp = lwarp[filtr]
        rwarp = rwarp[filtr]
        return super().forward(warpless, lwarp, rwarp, None)


class ExpWarp3Loss(ExpWarpLoss):
    def forward(self, actual, lwarp, rwarp):
        actual_P = actual[:, 1 - self._Pwarp]
        lwarp_P = lwarp[:, 1 - self._Pwarp]
        rwarp_P = rwarp[:, 1 - self._Pwarp]
        loss = (torch.abs(actual_P - lwarp_P) + torch.abs(actual_P - rwarp_P)) / 2
        return torch.mean(loss)


class ExpWarp6Loss(ExpWarpLoss):
    def __init__(self, yaw_warp, pitch_warp, target_mean=3, target_std=1):
        super().__init__(yaw_warp, pitch_warp)
        self._mean = target_mean
        self._std = target_std
        print(f'[{self.__class__.__name__}] Mean:{self._mean} Std:{self._std}')

    def forward(self, warpless, lwarp, rwarp, target):
        # For Yaw, lwarp > warpless > rwarp
        diff = torch.normal(self._mean, self._std, (len(target), )).to(target.device)
        # import pdb
        # pdb.set_trace()
        # warpless_ord = warpless[:, int(self._Pwarp)]
        # TODO: Check with shifted prediction as well
        lwarp_target = target[:, int(self._Pwarp)] + diff
        rwarp_target = target[:, int(self._Pwarp)] - diff

        lwarp_pred = lwarp[:, int(self._Pwarp)]
        rwarp_pred = rwarp[:, int(self._Pwarp)]
        loss_1 = torch.mean(torch.abs(lwarp_target - lwarp_pred))
        loss_2 = torch.mean(torch.abs(rwarp_target - rwarp_pred))

        # consistency loss
        warpless_cons = warpless[:, 1 - self._Pwarp]
        lwarp_cons = lwarp[:, 1 - self._Pwarp]
        rwarp_cons = rwarp[:, 1 - self._Pwarp]
        loss_3 = (torch.abs(warpless_cons - lwarp_cons) + torch.abs(warpless_cons - rwarp_cons)) / 2
        return (torch.mean(loss_1) + torch.mean(loss_2) + torch.mean(loss_3)) / 3
