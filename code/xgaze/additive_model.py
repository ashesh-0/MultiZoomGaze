import math

import torch.nn as nn

from static_model import GazeStaticModel


class AdditiveModel(GazeStaticModel):
    def get_latent_space(self, input):
        base_out = nn.ReLU()(self.base_model(input))
        output = self.last_layer(base_out).view(-1, self.output_dim)

        angular_output = output[:, :2]
        var = output[:, 2:3]
        return angular_output, var

    def forward(self, input):
        N, seq3, H, W = input.shape
        assert seq3 % 3 == 0
        seq = int(seq3 / 3)
        static_shape = (-1, 3) + input.size()[-2:]
        inp = input.view(static_shape)

        #
        angular_out, var = self.get_latent_space(inp)
        angular_out = angular_out.view(N, seq, angular_out.shape[-1])
        angular_out = angular_out.sum(dim=1)
        var = var.view(N, seq, var.shape[-1])
        var = var.sum(dim=1)

        # YAW
        angular_out[:, 0:1] = math.pi * nn.Tanh()(angular_out[:, 0:1])
        # Pitch
        angular_out[:, 1:2] = (math.pi / 2) * nn.Tanh()(angular_out[:, 1:2])
        var = math.pi * nn.Sigmoid()(var)
        var = var.view(-1, 1).expand(var.size(0), 2)
        return angular_out, var
