import math

import torch.nn as nn

from backbones.backbone_type import BackboneType
from backbones.interface import get_backbone


class GazeSinCosLSTM(nn.Module):
    """
    Here, we predict sin(yaw),cos(yaw),sin(pitch) and a variance signal. Same variance signal is broadcasted to all
    three
    """

    def __init__(
            self,
            fc2: int = 256,
            freeze_layer_idx: int = 0,
            seq_len=7,
            target_seq_index=None,
            backbone_type=BackboneType.Resnet18,
            bidirectional_lstm=True,
    ):
        """
        Args:
            freeze_layer_idx: all layers before this are not trainable for base model.
            target_seq_index: Which index of lstm output should be treated as the final feature which should then be fed
                into dense HEAD for prediction. By default, it is the middle element. So, for sequence length 7, it is
                3.
        """
        super(GazeSinCosLSTM, self).__init__()
        self.img_feature_dim = fc2  # the dimension of the CNN feature to represent each frame
        self._seq_len = seq_len
        self._target_seq_index = target_seq_index
        self._bidirectional_lstm = bidirectional_lstm
        if self._target_seq_index is None:
            self._target_seq_index = self._seq_len // 2

        assert self._target_seq_index < self._seq_len and self._target_seq_index >= 0

        self.base_model = get_backbone(backbone_type, pretrained=True, output_dim=self.img_feature_dim)

        self._freeze_layer_idx = freeze_layer_idx

        if self._freeze_layer_idx > 0:
            for i, nparam in enumerate(self.base_model.named_parameters()):
                name, param = nparam
                if i >= freeze_layer_idx:
                    print('Freezing layers before', name)
                    break
                param.requires_grad = False

        self.lstm = nn.LSTM(
            self.img_feature_dim,
            self.img_feature_dim,
            bidirectional=self._bidirectional_lstm,
            num_layers=2,
            batch_first=True)

        # The linear layer that maps the LSTM with the 4 outputs
        self.last_layer = nn.Linear((1 + int(self._bidirectional_lstm)) * self.img_feature_dim, 4)

        print(f'[{self.__class__.__name__}] Freeze:{self._freeze_layer_idx}'
              f' Bi-direc:{self._bidirectional_lstm} TargetIdx:{self._target_seq_index}')

    def unfreeze_all(self):
        print('[GazeSinCosLSTM] Unfreezing all layers')
        for param in self.base_model.parameters():
            param.requires_grad = True

    def get_lstm_output(self, input):
        self.lstm.flatten_parameters()
        static_shape = (-1, 3) + input.size()[-2:]
        inp = input.view(static_shape)
        base_out = self.base_model(inp)

        base_out = base_out.view(input.size(0), self._seq_len, self.img_feature_dim)

        lstm_out, _ = self.lstm(base_out)
        return lstm_out

    def get_aggregated_features(self, input):
        lstm_out = self.get_lstm_output(input)
        lstm_out = lstm_out[:, self._target_seq_index, :]
        return lstm_out

    def forward(self, input):
        agg_features = self.get_aggregated_features(input)
        output = self.last_layer(agg_features).view(-1, 4)

        angular_output = nn.Tanh()(output[:, :3])
        # NOTE: math.pi is not appropriate with sinecosine model.
        var = math.pi * nn.Sigmoid()(output[:, 3:])
        var = var.view(-1, 1).expand(var.size(0), 3)

        return angular_output, var
