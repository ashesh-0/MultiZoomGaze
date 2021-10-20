import numpy as np
import torch

from analysis.multicrop_analysis import MulticropContribution


class MulticropContributionNaive:
    def __init__(self):
        self._N = 0
        self._data = None

    def update(self, embeddings_2d):
        embeddings_2d = embeddings_2d.numpy()
        nBatch, nCrop, nChannel, H, W = embeddings_2d.shape
        if self._data is None:
            self._data = np.zeros((nCrop, nCrop, nChannel, H, W))

        index = np.argsort(embeddings_2d, axis=1)
        for nB in range(nBatch):
            for nCh in range(nChannel):
                for nH in range(H):
                    for nW in range(W):
                        for rank in range(nCrop):
                            value = index[nB, rank, nCh, nH, nW]
                            self._data[value, rank, nCh, nH, nW] += 1

        self._data = torch.Tensor(self._data)

    def get(self):
        return self._data


def test_vectorized_code_correctness():
    mc1 = MulticropContribution()
    mc2 = MulticropContributionNaive()
    emb = torch.Tensor(np.random.rand(32, 4, 512, 7, 7))
    mc1.update(emb)
    mc2.update(emb)
    assert np.all(mc1.get().numpy() == mc2.get().numpy())
    assert set(np.unique(np.sum(mc1.get().numpy(), axis=1))) == set([32])
