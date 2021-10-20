import torch.nn as nn

from backbones.backbone_type import BackboneType
from backbones.interface import get_backbone_cnn, get_backbone_head
from core.aggregation_module import AggregationModule
from core.aggregation_type import AggregationType
from sinecosine_model.model import GazeSinCosLSTM


class GazeSinCosMultiCropModel(GazeSinCosLSTM):
    def __init__(
        self,
        seq_len=7,
        img_feature_dim=256,
        backbone_type=BackboneType.Resnet18,
        atype: AggregationType = AggregationType.MAX,
        cropsize_list=None,
    ):
        super(GazeSinCosMultiCropModel, self).__init__(seq_len=seq_len,
                                                       fc2=img_feature_dim,
                                                       backbone_type=backbone_type)
        del self.lstm, self.last_layer
        assert seq_len == len(cropsize_list)
        self.lstm = None
        self._atype = atype
        self.base_model_tail = None
        if self._atype in [
                AggregationType.SPATIAL_MAX,
                AggregationType.SPATIAL_ATTENTION,
                AggregationType.SPATIAL_ATTENTION_ALIGNED,
                AggregationType.SPATIAL_MEAN,
        ]:
            self.base_model = get_backbone_cnn(backbone_type, pretrained=True)
            self.base_model_tail = get_backbone_head(backbone_type, pretrained=True, output_dim=self.img_feature_dim)

        self.last_layer = nn.Linear(self.img_feature_dim, 4)
        self.aggregator = AggregationModule(self.base_model,
                                            self.base_model_tail,
                                            atype,
                                            seq_len,
                                            img_feature_dim,
                                            cropsize_list=cropsize_list)

        # TODO: This is just done so that weights corresponding to previous model structure can load
        self._agg_module = self.aggregator.agg_module

    def get_aggregated_features(self, input):
        return self.aggregator.get_aggregated_features(input)
