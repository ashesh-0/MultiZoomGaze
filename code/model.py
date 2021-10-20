import math

import torch.nn as nn

from backbones.backbone_type import BackboneType
from backbones.interface import get_backbone


class GazeLSTM(nn.Module):
    def __init__(
            self,
            output_dim=3,
            seq_len=7,
            target_seq_index=None,
            backbone_type=BackboneType.Resnet18,
    ):
        super(GazeLSTM, self).__init__()
        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.target_seq_index = target_seq_index
        if self.target_seq_index is None:
            self.target_seq_index = self.seq_len // 2
        self.base_model = get_backbone(backbone_type, pretrained=True, output_dim=self.img_feature_dim)

        self.lstm = nn.LSTM(
            self.img_feature_dim, self.img_feature_dim, bidirectional=True, num_layers=2, batch_first=True)

        # The linear layer that maps the LSTM with the 3 outputs
        self.last_layer = nn.Linear(2 * self.img_feature_dim, self.output_dim)
        print(f'[{self.__class__.__name__} FDim:{self.img_feature_dim} ODim:{self.output_dim}'
              f' Seq:{self.seq_len} TIdx:{self.target_seq_index}')

    def get_aggregated_features(self, input):
        base_out = self.base_model(input.view((-1, 3) + input.size()[-2:]))
        base_out = base_out.view(input.size(0), self.seq_len, self.img_feature_dim)
        lstm_out, _ = self.lstm(base_out)
        lstm_out = lstm_out[:, self.target_seq_index, :]
        return lstm_out

    def forward(self, input):
        if self.lstm is not None:
            self.lstm.flatten_parameters()

        agg_feat = self.get_aggregated_features(input)
        output = self.last_layer(agg_feat).view(-1, self.output_dim)

        angular_output = output[:, :2]
        angular_output[:, 0:1] = math.pi * nn.Tanh()(angular_output[:, 0:1])
        angular_output[:, 1:2] = (math.pi / 2) * nn.Tanh()(angular_output[:, 1:2])
        if self.output_dim == 2:
            return angular_output
        assert self.output_dim == 3
        var = math.pi * nn.Sigmoid()(output[:, 2:3])
        var = var.view(-1, 1).expand(var.size(0), 2)

        return angular_output, var
