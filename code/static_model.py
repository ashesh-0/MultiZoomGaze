import math

import torch.nn as nn

from backbones.backbone_type import BackboneType
from backbones.interface import get_backbone


class GazeStaticModel(nn.Module):
    def __init__(self, output_dim=3, backbone_type=BackboneType.Resnet18):
        super(GazeStaticModel, self).__init__()
        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame
        self.output_dim = output_dim
        self.base_model = get_backbone(backbone_type, pretrained=True, output_dim=self.img_feature_dim)

        # The linear layer that maps the LSTM with the 3 or 2 outputs
        self.last_layer = nn.Linear(self.img_feature_dim, output_dim)

    def forward(self, input):
        base_out = nn.ReLU()(self.base_model(input))
        output = self.last_layer(base_out).view(-1, self.output_dim)

        angular_output = output[:, :2]
        # YAW
        angular_output[:, 0:1] = math.pi * nn.Tanh()(angular_output[:, 0:1])
        # Pitch
        angular_output[:, 1:2] = (math.pi / 2) * nn.Tanh()(angular_output[:, 1:2])
        if self.output_dim == 2:
            return angular_output
        assert self.output_dim == 3
        var = math.pi * nn.Sigmoid()(output[:, 2:3])
        var = var.view(-1, 1).expand(var.size(0), 2)

        return angular_output, var
