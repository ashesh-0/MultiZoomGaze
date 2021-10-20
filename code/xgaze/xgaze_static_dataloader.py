"""
Adapted from https://github.com/xucong-zhang/ETH-XGaze/blob/master/data_loader.py
"""
import os
import random
from typing import List

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from sinecosine_model.train_utils import get_sincos_gaze_from_spherical


class GazeDataset(Dataset):
    def __init__(self,
                 dataset_path: str,
                 keys_to_use: List[str] = None,
                 sub_folder='',
                 transform=None,
                 is_shuffle=True,
                 index_file=None,
                 sc_target=False,
                 is_load_label=True):
        self.path = dataset_path
        self.hdfs = {}
        self.sub_folder = sub_folder
        self.is_load_label = is_load_label
        self.sc_target = sc_target
        # assert len(set(keys_to_use) - set(all_keys)) == 0
        # Select keys
        # TODO: select only people with sufficient entries?
        self.selected_keys = [k for k in keys_to_use]
        assert len(self.selected_keys) > 0

        for num_i in range(0, len(self.selected_keys)):
            file_path = os.path.join(self.path, self.sub_folder, self.selected_keys[num_i])
            self.hdfs[num_i] = h5py.File(file_path, 'r', swmr=True)
            # print('read file: ', os.path.join(self.path, self.selected_keys[num_i]))
            assert self.hdfs[num_i].swmr_mode

        # Construct mapping from full-data index to key and person-specific index
        if index_file is None:
            self.idx_to_kv = []
            for num_i in range(0, len(self.selected_keys)):
                n = self.hdfs[num_i]["face_patch"].shape[0]
                self.idx_to_kv += [(num_i, i) for i in range(n)]
        else:
            print('load the file: ', index_file)
            self.idx_to_kv = np.loadtxt(index_file, dtype=np.int)

        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

        if is_shuffle:
            random.shuffle(self.idx_to_kv)  # random the order to stable the training

        self.transform = transform
        print(f'[{self.__class__.__name__}] SC:{int(self.sc_target)} Keys:{len(keys_to_use)}')

    def __len__(self):
        return len(self.idx_to_kv)

    def __del__(self):
        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

    def _get_img(self, hdf, hdf_idx):
        # Get face image
        image = hdf['face_patch'][hdf_idx, :]
        image = image[:, :, [2, 1, 0]]  # from BGR to RGB
        transformed_image = self.transform(image)
        return {'raw': image, 'img': transformed_image}

    def get_gaze(self, idx):
        key, idx = self.idx_to_kv[idx]
        hdf = h5py.File(os.path.join(self.path, self.sub_folder, self.selected_keys[key]), 'r', swmr=True)
        assert hdf.swmr_mode
        return self._get_gaze(hdf, idx)

    def _get_gaze(self, hdf, idx):
        gaze_label = hdf['face_gaze'][idx, :]
        gaze_label = gaze_label.astype('float32')
        # NOTE: yaw,pitch is our ordering and so gaze_label is reordered. for test, we will need to predict
        # and reverse the prediction before submitting
        gaze_label = gaze_label[::-1].copy()
        if self.sc_target:
            gaze_label = get_sincos_gaze_from_spherical(torch.Tensor(gaze_label)).numpy()
        return gaze_label

    def __getitem__(self, idx):
        key, idx = self.idx_to_kv[idx]
        hdf = h5py.File(os.path.join(self.path, self.sub_folder, self.selected_keys[key]), 'r', swmr=True)
        assert hdf.swmr_mode

        # Get face image
        image = self._get_img(hdf, idx)['img']

        # Get labels
        if self.is_load_label:
            return image, self._get_gaze(hdf, idx)
        else:
            return image


class GazeDatasetWithId(GazeDataset):
    def __getitem__(self, idx):
        output = super().__getitem__(idx)
        target_id, _ = self.idx_to_kv[idx]
        if isinstance(output, tuple):
            return (*output, target_id)
        else:
            return (output, target_id)


class GazeDatasetCL(GazeDataset):
    BOTH = -1
    LEFT = 0
    RIGHT = 1

    def __init__(self,
                 dataset_path: str,
                 keys_to_use: List[str] = None,
                 sub_folder='',
                 transform=None,
                 is_shuffle=True,
                 index_file=None,
                 sc_target=False,
                 is_load_label=True):
        super().__init__(
            dataset_path,
            keys_to_use=keys_to_use,
            sub_folder=sub_folder,
            transform=transform,
            is_shuffle=is_shuffle,
            index_file=index_file,
            sc_target=sc_target,
            is_load_label=is_load_label,
        )
        self._leftright = GazeDatasetCL.LEFT

    def set_leftright(self, mode):
        self._leftright = mode
        if self._leftright == GazeDatasetCL.LEFT:
            print(f'[{self.__class__.__name__}] Setting leftright to LEFT')
        elif self._leftright == GazeDatasetCL.RIGHT:
            print(f'[{self.__class__.__name__}] Setting leftright to RIGHT')
        elif self._leftright == GazeDatasetCL.BOTH:
            print(f'[{self.__class__.__name__}] Setting leftright to BOTH')

    def _clip(self, img):
        if self._leftright in [GazeDatasetCL.LEFT, GazeDatasetCL.RIGHT]:
            if np.random.rand() >= 0.5:
                return img[:, :, :112]
            else:
                return img[:, :, 112:]
        elif self._leftright == GazeDatasetCL.BOTH:
            return img

    def __getitem__(self, idx):
        output = super().__getitem__(idx)
        if isinstance(output, tuple):
            assert len(output) == 2
            output = (self._clip(output[0]), output[1])
        else:
            output = self._clip(output)
        return output
