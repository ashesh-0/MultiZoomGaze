from core.aggregation_module import AggregationModule
from core.aggregation_type import AggregationType
from non_lstm_based_model import GazeMultiCropModel


class Embedding2DAggregation(AggregationModule):
    def get_embeddings(self, input):
        static_shape = (-1, 3) + input.size()[-2:]
        inp = input.view(static_shape)
        base_out = self.base_model(inp)

        f_len = base_out.shape[1]
        assert self._atype in [
            AggregationType.SPATIAL_MAX, AggregationType.SPATIAL_ATTENTION, AggregationType.SPATIAL_ATTENTION_ALIGNED,
            AggregationType.SPATIAL_ATTENTION_V2, AggregationType.SPATIAL_MEAN
        ]
        sz = base_out.shape[-1]
        base_out = base_out.view(input.size(0), self._seq_len, f_len, sz, sz)
        return base_out


class Embedding2DModel(GazeMultiCropModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aggregator = Embedding2DAggregation(self.base_model,
                                                 self.base_model_tail,
                                                 self.aggregator._atype,
                                                 self.seq_len,
                                                 self.img_feature_dim,
                                                 cropsize_list=self.aggregator._cropsize_list)

    def forward(self, input):
        return self.get_aggregated_features(input)
