import numpy as np
import torch


class MulticropContribution:
    def __init__(self, agg_dimensions=(0, )):
        self._N = 0
        self._data = None
        assert isinstance(agg_dimensions, tuple)
        self._agg_dimensions = agg_dimensions
        assert 1 not in self._agg_dimensions, 'We want to differentiate on cropsize'

    def _init_data(self, embedding_shape):
        """
        Embedding has following shape: (nBatch, nCrop, nChannel, H, W)
        """
        nBatch, nCrop, nChannel, H, W = embedding_shape
        nRank = nCrop
        assert self._data is None
        out_dims = []
        if 0 not in self._agg_dimensions:
            out_dims.append(nBatch)

        out_dims += [nCrop, nRank]

        if 2 not in self._agg_dimensions:
            out_dims.append(nChannel)
        if 3 not in self._agg_dimensions:
            out_dims.append(H)
        if 4 not in self._agg_dimensions:
            out_dims.append(W)
        self._data = torch.zeros(tuple(out_dims))

    def update(self, embeddings_2d):
        nBatch, nCrop, nChannel, H, W = embeddings_2d.shape
        nRank = nCrop
        if self._data is None:
            self._init_data(embeddings_2d.shape)
            self._data = self._data.to(embeddings_2d.device)

        ordering = torch.argsort(embeddings_2d, axis=1)
        for rank in range(nRank):
            for crop in range(nCrop):
                crop_rank = ordering[:, rank, :, :, :] == crop
                agg_axis = np.array(self._agg_dimensions)
                # When computing crop_rank, dimensions have gotten reduced by 1.
                agg_axis -= 1
                agg_axis[agg_axis < 0] = 0
                agg_axis = tuple(agg_axis)
                self._data[crop, rank] += torch.sum(crop_rank, dim=agg_axis)

        self._N += nBatch

    def get(self):
        return self._data


class MulticropContributionPerSample(MulticropContribution):
    def __init__(self, dataset_size, agg_dimensions=(2, 3, 4)):
        super().__init__(agg_dimensions=agg_dimensions)
        assert 0 not in self._agg_dimensions, 'We want to have an entry for every entry in dataset. So batch dimension \
            should not be aggregated.'

        self._dataset_size = dataset_size

    def update(self, embeddings_2d):
        nBatch, nCrop, nChannel, H, W = embeddings_2d.shape
        nRank = nCrop
        if self._data is None:
            self._init_data((self._dataset_size, *embeddings_2d.shape[1:]))
            self._data = self._data.to(embeddings_2d.device)

        ordering = torch.argsort(embeddings_2d, axis=1)
        for rank in range(nRank):
            for crop in range(nCrop):
                crop_rank = ordering[:, rank, :, :, :] == crop
                agg_axis = np.array(self._agg_dimensions)
                # When computing crop_rank, dimensions have gotten reduced by 1.
                assert 0 not in agg_axis
                agg_axis -= 1
                agg_axis = tuple(agg_axis)
                # import pdb
                # pdb.set_trace()
                self._data[self._N:self._N + nBatch, crop, rank] += torch.sum(crop_rank, axis=agg_axis)

        self._N += nBatch

    def get(self):
        return self._data
