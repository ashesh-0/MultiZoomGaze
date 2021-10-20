import torch.nn as nn

from backbones.backbone_type import BackboneType
from backbones.interface import get_backbone_cnn, get_backbone_head
from core.aggregation_module import AggregationModule
from core.aggregation_type import AggregationType
from model import GazeLSTM


class GazeMultiCropModel(GazeLSTM):
    def __init__(
        self,
        output_dim=3,
        backbone_type=BackboneType.Resnet18,
        atype: AggregationType = AggregationType.SPATIAL_MAX,
        cropsize_list=None,
    ):
        super(GazeMultiCropModel, self).__init__(seq_len=len(cropsize_list),
                                                 output_dim=output_dim,
                                                 backbone_type=backbone_type)
        del self.lstm, self.last_layer
        self.lstm = None
        self._atype = atype
        self.base_model_tail = None
        if self._atype in [
                AggregationType.SPATIAL_MAX,
                AggregationType.SPATIAL_ATTENTION,
                AggregationType.SPATIAL_ATTENTION_ALIGNED,
                AggregationType.SPATIAL_ATTENTION_V2,
                AggregationType.SPATIAL_MEAN,
        ]:
            self.base_model = get_backbone_cnn(backbone_type, pretrained=True)
            self.base_model_tail = get_backbone_head(backbone_type, pretrained=True, output_dim=self.img_feature_dim)

        self.last_layer = nn.Linear(self.img_feature_dim, self.output_dim)
        self.aggregator = AggregationModule(self.base_model,
                                            self.base_model_tail,
                                            atype,
                                            self.seq_len,
                                            self.img_feature_dim,
                                            cropsize_list=cropsize_list)

        # TODO: This is just done so that weights corresponding to previous model structure can load
        self._agg_module = self.aggregator.agg_module

    def get_aggregated_features(self, input):
        return self.aggregator.get_aggregated_features(input)
