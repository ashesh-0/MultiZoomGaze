import math

import torch.nn as nn

from static_model import GazeStaticModel


class AdverserialBranch(nn.Module):
    def __init__(
        self,
        adv_N=65,
    ):
        super(AdverserialBranch, self).__init__()
        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame
        self._adv_N = adv_N
        self.adv_layer = nn.Sequential(
            nn.Linear(self.img_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self._adv_N),
        )

    def forward(self, input):
        return self.adv_layer(input)


class GazeStaticAdverserialModel(GazeStaticModel):
    def forward(self, input):
        base_out = nn.ReLU()(self.base_model(input))
        output = self.last_layer(base_out).view(-1, self.output_dim)

        angular_output = output[:, :2]
        # YAW
        angular_output[:, 0:1] = math.pi * nn.Tanh()(angular_output[:, 0:1])
        # Pitch
        angular_output[:, 1:2] = (math.pi / 2) * nn.Tanh()(angular_output[:, 1:2])
        assert self.output_dim == 3
        var = math.pi * nn.Sigmoid()(output[:, 2:3])
        var = var.view(-1, 1).expand(var.size(0), 2)

        return angular_output, var, base_out
