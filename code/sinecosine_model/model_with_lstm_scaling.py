import math

import torch
import torch.nn as nn

from core.diff_crop_layer import DiffCropOneImage
from backbones.resnet import ViewDense, resnet18

# from stn.spatial_transformer import SpatialTransformer


class GazeSinCosLSTMLstmScaling(nn.Module):
    """
    Here, we predict sin(yaw),cos(yaw),sin(pitch) and a variance signal. Same variance signal is broadcasted to all
    three
    """

    def __init__(self, seq_len: int, scaling_dict, fc2: int = 256):
        """
        Args:
            freeze_layer_idx: all layers before this are not trainable for base model.
        """
        super(GazeSinCosLSTMLstmScaling, self).__init__()
        assert seq_len % 2 == 1
        for t in range(seq_len):
            assert t in scaling_dict
            for i in range(1, len(scaling_dict[t])):
                assert scaling_dict[t][i] <= scaling_dict[t][i - 1]

        self._seq_len = seq_len
        self._scales = scaling_dict

        self.img_feature_dim = fc2  # the dimension of the CNN feature to represent each frame

        # self.base_model = nn.Sequential(
        #     # nn.AvgPool2d(2),
        #     nn.Conv2d(3, 32, kernel_size=3),
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 32, kernel_size=3),
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.Conv2d(32, 64, kernel_size=3),
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 64, kernel_size=3),
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.Conv2d(64, 128, kernel_size=3),
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.Conv2d(128, self.img_feature_dim, kernel_size=3),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True),
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     ViewDense(),
        # )

        self.base_model = resnet18(pretrained=True)
        self.base_model.fc2 = nn.Linear(1000, self.img_feature_dim)

        self.lstm_attention = nn.LSTM(
            self.img_feature_dim, self.img_feature_dim // 2, bidirectional=True, num_layers=2, batch_first=True)

        self.lstm = nn.LSTM(
            self.img_feature_dim, self.img_feature_dim, bidirectional=True, num_layers=2, batch_first=True)

        # The linear layer that maps the LSTM with the 4 outputs
        self.last_layer = nn.Linear(2 * self.img_feature_dim, 4)

        self._scales_layer_dict = nn.ModuleList([DiffCropOneImage(self._scales[t]) for t in range(self._seq_len)])

        print(f'[{self.__class__.__name__}] seq_len:{self._seq_len} fc2:{fc2} ScalingDict:{scaling_dict}')

    def scale_invariant_features_V2(self, input):
        static_shape = (-1, 3) + input.size()[-2:]
        scales = self._scales[0]
        input_as_static = input.view(static_shape)
        # N*T,T_attention,C,H,W
        base_model_input = self._scales_layer_dict[0](input_as_static)
        # N*T*T_attention, img_feature_dim
        base_model_output = self.base_model(base_model_input.view(static_shape))
        attention_T = len(scales)
        lstm_attention_shape = (-1, attention_T, self.img_feature_dim)
        # N*T,T_attention,img_feature_dim
        lstm_input = base_model_output.view(lstm_attention_shape)
        lstm_output, _ = self.lstm_attention(lstm_input)
        # N*T,img_feature_dim
        lstm_output = lstm_output[:, lstm_input.shape[1] // 2, :]
        return lstm_output.view((input.size(0), self._seq_len, self.img_feature_dim))

    # def scale_invariant_features(self, input):
    #     lstm_shape = ((input.size(0), self._seq_len, 3) + input.size()[-2:])
    #     static_shape = (-1, 3) + input.size()[-2:]
    #     features = []
    #     input_as_lstm = input.view(lstm_shape)
    #     for t in range(self._seq_len):
    #         base_model_input = self._scales_layer_dict[t](input_as_lstm[:, t, ...])
    #         base_model_output = self.base_model(base_model_input.view(static_shape))
    #         attention_T = len(self._scales[t])
    #         lstm_attention_shape = (input.shape[0], attention_T, self.img_feature_dim)
    #         lstm_input = base_model_output.view(lstm_attention_shape)
    #         # print(t, self._scales[t], lstm_input.shape)
    #         # import pdb
    #         # pdb.set_trace()
    #         lstm_output, _ = self.lstm_attention(lstm_input)
    #         # NOTE: check here for lstm_input.shape
    #         lstm_output = lstm_output[:, lstm_input.shape[1] // 2, :]
    #         features.append(lstm_output[:, None, ...])

    #     return torch.cat(features, dim=1)

    def forward(self, input):
        self.lstm.flatten_parameters()
        self.lstm_attention.flatten_parameters()
        # import pdb
        # pdb.set_trace()
        scale_inv_inp = self.scale_invariant_features_V2(input)
        # scale_inv_inp = self.scale_invariant_features(input)
        # temp = self.base_model(input.view(-1, 3, 224, 224))
        # scale_inv_inp = temp.view((input.size(0), self._seq_len, -1))
        lstm_out, _ = self.lstm(scale_inv_inp)
        # import pdb
        # pdb.set_trace()
        lstm_out = lstm_out[:, self._seq_len // 2, :]
        output = self.last_layer(lstm_out).view(-1, 4)

        angular_output = nn.Tanh()(output[:, :3])

        var = math.pi * nn.Sigmoid()(output[:, 3:])
        var = var.view(-1, 1).expand(var.size(0), 3)

        return angular_output, var


class GazeSinCosLSTMLstmScalingV2(nn.Module):
    """
    """

    def __init__(self, seq_len: int, num_scales, fc2: int = 256):
        """
        Args:
            freeze_layer_idx: all layers before this are not trainable for base model.
        """
        super(GazeSinCosLSTMLstmScalingV2, self).__init__()
        assert seq_len % 2 == 1
        self._num_scales = num_scales
        self._seq_len = seq_len

        self.img_feature_dim = fc2  # the dimension of the CNN feature to represent each frame

        self.base_model = resnet18(pretrained=True)
        self.base_model.fc2 = nn.Linear(1000, self.img_feature_dim)

        self.lstm_attention = nn.LSTM(
            self.img_feature_dim, self.img_feature_dim // 2, bidirectional=True, num_layers=2, batch_first=True)

        self.lstm = nn.LSTM(
            self.img_feature_dim, self.img_feature_dim, bidirectional=True, num_layers=2, batch_first=True)

        # The linear layer that maps the LSTM with the 4 outputs
        self.last_layer = nn.Linear(2 * self.img_feature_dim, 4)

        print(f'[{self.__class__.__name__}] seq_len:{self._seq_len} fc2:{fc2} NumScales:{self._num_scales}')

    def scale_invariant_features(self, input):
        # input.shape (N,T*T_attention*3,sz,sz)

        static_shape = (-1, 3) + input.size()[-2:]
        # N*T*T_attention, img_feature_dim
        base_model_output = self.base_model(input.view(static_shape))
        lstm_attention_shape = (-1, self._num_scales, self.img_feature_dim)
        # N*T,T_attention,img_feature_dim
        lstm_input = base_model_output.view(lstm_attention_shape)
        lstm_output, _ = self.lstm_attention(lstm_input)
        # N*T,img_feature_dim
        lstm_output = lstm_output[:, lstm_input.shape[1] // 2, :]
        return lstm_output.view((input.size(0), self._seq_len, self.img_feature_dim))

    def forward(self, input):
        self.lstm.flatten_parameters()
        self.lstm_attention.flatten_parameters()
        scale_inv_inp = self.scale_invariant_features(input)
        lstm_out, _ = self.lstm(scale_inv_inp)
        lstm_out = lstm_out[:, self._seq_len // 2, :]
        output = self.last_layer(lstm_out).view(-1, 4)

        angular_output = nn.Tanh()(output[:, :3])

        var = math.pi * nn.Sigmoid()(output[:, 3:])
        var = var.view(-1, 1).expand(var.size(0), 3)

        return angular_output, var
