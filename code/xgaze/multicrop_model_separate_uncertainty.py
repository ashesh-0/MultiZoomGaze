import math

import torch.nn as nn

from backbones.backbone_type import BackboneType
from core.aggregation_type import AggregationType
from non_lstm_based_model import GazeMultiCropModel


class GazeMultiCropSepModel(GazeMultiCropModel):
    def __init__(
        self,
        backbone_type=BackboneType.Resnet18,
        atype: AggregationType = AggregationType.SPATIAL_MAX,
        cropsize_list=None,
    ):
        super().__init__(
            output_dim=4,
            backbone_type=backbone_type,
            atype=atype,
            cropsize_list=cropsize_list,
        )

    def forward(self, input):
        assert self.lstm is None

        agg_feat = self.get_aggregated_features(input)
        output = self.last_layer(agg_feat).view(-1, self.output_dim)

        angular_output = output[:, :2]
        angular_output[:, 0:1] = math.pi * nn.Tanh()(angular_output[:, 0:1])
        angular_output[:, 1:2] = (math.pi / 2) * nn.Tanh()(angular_output[:, 1:2])
        assert self.output_dim == 4
        var = math.pi * nn.Sigmoid()(output[:, 2:4])

        return angular_output, var
