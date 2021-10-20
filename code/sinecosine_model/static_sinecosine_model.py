import torch
import torch.nn as nn

from backbones.backbone_type import BackboneType
from backbones.interface import get_backbone
from backbones.weight_utils import freeze_backbone_weights, load_backbone_weights


class GazeStaticSineAndCosineModelRaw(nn.Module):
    """
    Here, we predict sin(yaw), cos(yaw),sin(pitch) and a variance signal. Same variance signal is broadcasted to
    all three.
    """

    def __init__(self,
                 fc1=None,
                 fc2: int = 256,
                 freeze_layer_idx: int = 0,
                 backbone_loader_checkpoint_fpath=None,
                 output_dim=4,
                 stn_pre_load=False,
                 backbone_type=BackboneType.Resnet18):
        """
        Args:
            freeze_layer_idx: all layers before this are not trainable for base model.
        """
        super(GazeStaticSineAndCosineModelRaw, self).__init__()
        self.img_feature_dim = fc2  # the dimension of the CNN feature to represent each frame
        self._freeze_layer_idx = freeze_layer_idx
        self._output_dim = output_dim
        self._backbone_loader_checkpoint_fpath = backbone_loader_checkpoint_fpath

        assert fc1 is None, 'Support for fc1 is deprecated'
        self.base_model = get_backbone(backbone_type, pretrained=True, output_dim=self.img_feature_dim)
        if self._backbone_loader_checkpoint_fpath is not None:
            load_backbone_weights(self, self._backbone_loader_checkpoint_fpath)

        if self._freeze_layer_idx > 0:
            freeze_backbone_weights(self, self._freeze_layer_idx)

        self.last_layer = nn.Linear(self.img_feature_dim, self._output_dim)
        print(f'[StaticSinCosModel] fc1:{fc1} fc2:{fc2} OutDim:{self._output_dim}')

    def forward(self, input):
        base_out = nn.ReLU()(self.base_model(input))
        return self.last_layer(base_out)

    def unfreeze_all(self):
        print('[GazeStaticSineAndCosineModelRaw] Unfreezing all layers')
        for param in self.base_model.parameters():
            param.requires_grad = True


class GazeStaticSineAndCosineModel(GazeStaticSineAndCosineModelRaw):
    def forward(self, input):
        output = super().forward(input).view(-1, 4)

        # [sin(Yaw),cos(Yaw),sin(Pitch)]
        angular_output = nn.Tanh()(output[:, :3])
        var = nn.Sigmoid()(output[:, 3:])
        var = var.view(-1, 1).expand(var.size(0), 3)
        return angular_output, var


class GazeStaticMultiSineAndCosineModel(GazeStaticSineAndCosineModelRaw):
    def __init__(self, fc2: int = 256):
        super().__init__(fc2=fc2, output_dim=6)

    def forward(self, input):
        output = super().forward(input).view(-1, self._output_dim)

        # [sin(Yaw),cos(Yaw),..,sin(Pitch)]
        angular_output = nn.Tanh()(output[:, :self._output_dim - 1])
        var = nn.Sigmoid()(output[:, self._output_dim - 1:])
        var = var.view(-1, 1).expand(var.size(0), self._output_dim - 1)
        return angular_output, var


if __name__ == '__main__':
    from core.analysis_utils import count_parametersM
    print(f'{count_parametersM(GazeStaticSineAndCosineModel())}M parameters')
