import torch
import torch.nn as nn

from backbones.backbone_type import BackboneType
from backbones.interface import get_backbone
from backbones.resnet import get_resnet_parts
from backbones.weight_utils import freeze_backbone_weights, load_backbone_weights


class LazyAggregationModel(nn.Module):
    """
    Here, we predict sin(yaw), cos(yaw),sin(pitch) and a variance signal. Aggregation is done at
     intemediate feature map level.
    """
    def __init__(self,
                 fc1=None,
                 fc2: int = 256,
                 agg_layer_idx=None,
                 backbone_loader_checkpoint_fpath=None,
                 output_dim=4,
                 stn_pre_load=False,
                 backbone_type=BackboneType.Resnet18):
        """
        Args:
            agg_layer_idx: aggregation happens at this layer.
        """
        super(LazyAggregationModel, self).__init__()
        self.img_feature_dim = fc2  # the dimension of the CNN feature to represent each frame
        self._cropsize_list = [224, 200, 175, 150]
        self._img_size = 224
        self._output_dim = output_dim
        self._backbone_loader_checkpoint_fpath = backbone_loader_checkpoint_fpath
        self._agg_layer_idx = agg_layer_idx

        assert backbone_type == BackboneType.Resnet18, 'Only backbone supported is Resnet'
        assert fc1 is None, 'Support for fc1 is deprecated'
        assert agg_layer_idx is not None and isinstance(agg_layer_idx, int), f'Invalid layer_idx:{agg_layer_idx}'
        assert self._backbone_loader_checkpoint_fpath is None

        self.model1, self.model2 = get_resnet_parts(self.img_feature_dim, part1_end_idx=agg_layer_idx)
        self.last_layer = nn.Linear(self.img_feature_dim, self._output_dim)
        print(f'[{self.__class__.__name__}] OutDim:{self._output_dim} Cropsizes:{self._cropsize_list}'
              f' AggIdx:{agg_layer_idx}')

    def _aggregate(self, feature_map):
        _, _, H, W = feature_map.shape
        # NOTE: Check that channel comes first, then H,w
        assert H == W, 'It can be implemented but is enforced for convenience.'
        aggregated = 0
        for cropsize in self._cropsize_list:
            new_H = int(H * cropsize / self._img_size)
            l_pad = (H - new_H) // 2
            r_pad = H - new_H - l_pad
            if r_pad == 0:
                aggregated += feature_map / len(self._cropsize_list)
                continue

            assert r_pad > 0
            # NOTE: channel ordering will affect here as well.
            zoomed_in_feature_map = feature_map[:, :, l_pad:-r_pad, l_pad:-r_pad]
            aggregated += nn.Upsample(size=H)(zoomed_in_feature_map) / len(self._cropsize_list)
        return aggregated

    def forward(self, input):
        feature_map = self.model1(input)
        aggregated_map = self._aggregate(feature_map)

        base_out = nn.ReLU()(self.model2(aggregated_map))
        output = self.last_layer(base_out).view(-1, 4)
        angular_output = nn.Tanh()(output[:, :3])
        var = nn.Sigmoid()(output[:, 3:])
        var = var.view(-1, 1).expand(var.size(0), 3)
        return angular_output, var

    def unfreeze_all(self):
        print(f'[{self.__class__.__name__}] Unfreezing all layers')
        for param in self.base_model.parameters():
            param.requires_grad = True
