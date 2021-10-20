import torch
import torch.nn as nn


class DiffCropOneImage(nn.Module):
    def __init__(self, cropsize_list, upsample_mode='nearest'):
        super().__init__()
        self._c_list = cropsize_list
        self._mode = upsample_mode

    def forward(self, input):
        N, C, W, H = input.shape
        outputs = []
        upsampler = nn.Upsample(size=(W, H), mode=self._mode)
        for sz in self._c_list:
            h_s = int(H / 2 - sz / 2)
            h_e = int(H / 2 + sz / 2)
            w_s = int(W / 2 - sz / 2)
            w_e = int(W / 2 + sz / 2)
            outputs.append(upsampler(input[:, :, h_s:h_e, w_s:w_e])[:, None, :, :, :])

        output = torch.cat(outputs, dim=1)
        # print('DiffcropOneImage', self._c_list)
        return output


class DiffCrop(nn.Module):
    def __init__(self, cropsize_list, upsample_mode='nearest'):
        super().__init__()
        self._c_list = cropsize_list
        self._mode = upsample_mode

    def forward(self, input):
        N, T, C, W, H = input.shape
        assert len(self._c_list) == T
        outputs = []
        for t in range(T):
            upsampler = nn.Upsample(size=(W, H), mode=self._mode)
            h_s = int(H / 2 - self._c_list[t] / 2)
            h_e = int(H / 2 + self._c_list[t] / 2)
            w_s = int(W / 2 - self._c_list[t] / 2)
            w_e = int(W / 2 + self._c_list[t] / 2)

            outputs.append(upsampler(input[:, t, :, h_s:h_e, w_s:w_e])[:, None, :, :, :])

        output = torch.cat(outputs, dim=1)
        return output
