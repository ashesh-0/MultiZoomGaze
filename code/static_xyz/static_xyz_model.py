import math

import torch.nn as nn

from backbones.resnet import resnet18


class GazeStaticXyzModel(nn.Module):
    def __init__(self):
        super(GazeStaticXyzModel, self).__init__()
        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame

        self.base_model = resnet18(pretrained=True)

        self.base_model.fc2 = nn.Linear(1000, self.img_feature_dim)

        # The linear layer that maps the LSTM with the 4 outputs (x,y,z,var)
        self.last_layer = nn.Linear(self.img_feature_dim, 4)

    def forward(self, input):
        base_out = nn.ReLU()(self.base_model(input))
        output = self.last_layer(base_out).view(-1, 4)
        xyz = nn.Tanh()(output[:, :3])

        var = math.pi * nn.Sigmoid()(output[:, -1:])
        var = var.view(-1, 1).expand(var.size(0), 3)

        return xyz, var
