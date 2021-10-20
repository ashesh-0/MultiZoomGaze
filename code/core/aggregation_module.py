import torch
import torch.nn as nn

from core.aggregation_type import AggregationType
from core.attention import (AttentionMultiScale, SpatialAttentionMultiScale, SpatialAttentionMultiScaleAligned,
                            SpatialAttentionMultiScaleV2)


class SpatialMaxModule(nn.Module):
    def forward(self, input):
        return input.max(1)[0]


class SpatialMeanModule(nn.Module):
    def forward(self, input):
        return input.mean(1)


class AggregationModule:
    def __init__(self, base_model, base_model_tail, aggregation_type, seq_len, img_feature_dim, cropsize_list=None):
        self._atype = aggregation_type
        self.agg_module = None
        self.base_model = base_model
        self.base_model_tail = base_model_tail
        self._seq_len = seq_len
        self.img_feature_dim = img_feature_dim
        self._cropsize_list = cropsize_list

        if self._atype == AggregationType.SPATIAL_MAX:
            self.agg_module = SpatialMaxModule()
        if self._atype == AggregationType.SPATIAL_MEAN:
            self.agg_module = SpatialMeanModule()
        elif self._atype == AggregationType.SPATIAL_ATTENTION:
            self.agg_module = SpatialAttentionMultiScale()
        elif self._atype == AggregationType.SPATIAL_ATTENTION_ALIGNED:
            self.agg_module = SpatialAttentionMultiScaleAligned(cropsize_list)
        elif self._atype == AggregationType.SPATIAL_ATTENTION_V2:
            self.agg_module = SpatialAttentionMultiScaleV2()
        elif self._atype == AggregationType.ATTENTION:
            self.agg_module = AttentionMultiScale(fc=img_feature_dim)
        print(f'[{self.__class__.__name__}] AggregationMode:{AggregationType.name(self._atype)}')

    def get_embeddings(self, input):
        static_shape = (-1, 3) + input.size()[-2:]
        inp = input.view(static_shape)
        base_out = self.base_model(inp)

        f_len = base_out.shape[1]
        if self._atype in [
                AggregationType.SPATIAL_MAX, AggregationType.SPATIAL_ATTENTION,
                AggregationType.SPATIAL_ATTENTION_ALIGNED, AggregationType.SPATIAL_ATTENTION_V2,
                AggregationType.SPATIAL_MEAN
        ]:
            sz = base_out.shape[-1]
            base_out = base_out.view(input.size(0), self._seq_len, f_len, sz, sz)
            base_out = self.agg_module(base_out)
            base_out = self.base_model_tail(base_out)
            return base_out

        elif self._atype == AggregationType.ATTENTION:
            emb = base_out.view(input.size(0), self._seq_len, self.img_feature_dim)
            base_out = self.agg_module(input, emb)
            return base_out
        else:
            base_out = base_out.view(input.size(0), self._seq_len, self.img_feature_dim)
            return base_out

    def aggregate_embeddings(self, embeddings: torch.Tensor):
        if self._atype == AggregationType.MAX:
            return embeddings.max(1)[0]
        elif self._atype == AggregationType.MEAN:
            return torch.mean(embeddings, dim=1)

        elif self._atype in [
                AggregationType.SPATIAL_MAX, AggregationType.SPATIAL_ATTENTION,
                AggregationType.SPATIAL_ATTENTION_ALIGNED, AggregationType.SPATIAL_ATTENTION_V2,
                AggregationType.ATTENTION, AggregationType.SPATIAL_MEAN
        ]:
            # NOTE: Spatial aggregation has happened inside get_embeddings. No aggregation to be done here.
            return embeddings

    def get_aggregated_features(self, input):
        embeddings = self.get_embeddings(input)
        emb_out = self.aggregate_embeddings(embeddings)
        return emb_out
