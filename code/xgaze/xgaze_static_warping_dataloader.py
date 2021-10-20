import os
from typing import List

import h5py
import numpy as np
import torch
from torchvision import transforms

from sinecosine_model.train_utils import get_sincos_gaze_from_spherical
from xgaze.warp_utils import Warper, WarperWithCameraMatrixFull
from xgaze.xgaze_static_dataloader import GazeDataset


class GazeDatasetWithWarping(GazeDataset):
    def __init__(self,
                 dataset_path: str,
                 yaw_warping_angle: float,
                 pitch_warping_angle: float,
                 keys_to_use: List[str] = None,
                 sub_folder='',
                 transform=None,
                 is_shuffle=True,
                 index_file=None,
                 sc_target=False,
                 is_load_label=True):
        super().__init__(dataset_path,
                         keys_to_use=keys_to_use,
                         sub_folder=sub_folder,
                         transform=transform,
                         is_shuffle=is_shuffle,
                         index_file=index_file,
                         sc_target=sc_target,
                         is_load_label=is_load_label)
        self._yaw_angle = yaw_warping_angle
        self._pitch_angle = pitch_warping_angle
        assert self._yaw_angle == 0 or self._pitch_angle == 0
        focal_norm = 960  # focal length of normalized camera
        C = np.array([
            [focal_norm, 0, 112],
            [0, focal_norm, 112],
            [0, 0, 1.0],
        ])

        self.positive_warp = Warper(self._yaw_angle, self._pitch_angle)
        self.negative_warp = Warper(-self._yaw_angle, -1 * self._pitch_angle)
        self.resize_transform = transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize((224, 224)),
             transforms.ToTensor()])
        print(f'[{self.__class__.__name__}] Ywarp:{self._yaw_angle} Pwarp:{self._pitch_angle}')

    def _get_warped_img(self, img):
        pos = self.positive_warp.transform(img)
        sx, sy, ex, ey = self.positive_warp.boundaries
        pos = self.transform(self.resize_transform(pos[sy:ey, sx:ex]))[None, ...]

        neg = self.negative_warp.transform(img)
        sx, sy, ex, ey = self.negative_warp.boundaries
        neg = self.transform(self.resize_transform(neg[sy:ey, sx:ex]))[None, ...]

        # neg = self.transform(self.negative_warp.transform(img))[None, ...]
        return np.concatenate([pos, neg], axis=0)

    def __getitem__(self, idx):
        key, idx = self.idx_to_kv[idx]
        hdf = h5py.File(os.path.join(self.path, self.sub_folder, self.selected_keys[key]), 'r', swmr=True)
        assert hdf.swmr_mode

        # Get face image
        img_data = self._get_img(hdf, idx)
        image = img_data['img']

        # Get labels
        if self.is_load_label:
            warped_imgs = self._get_warped_img(img_data['raw'])
            gaze_label = hdf['face_gaze'][idx, :]
            gaze_label = gaze_label.astype('float32')
            # NOTE: yaw,pitch is our ordering and so gaze_label is reordered. for test, we will need to predict
            # and reverse the prediction before submitting
            gaze_label = gaze_label[::-1].copy()
            if self.sc_target:
                gaze_label = get_sincos_gaze_from_spherical(torch.Tensor(gaze_label)).numpy()
            return np.concatenate([image[None, ...], warped_imgs], axis=0), gaze_label
        else:
            return image


class GazeDatasetWithWarpingShort(GazeDatasetWithWarping):
    def __len__(self):
        return len(self.idx_to_kv) // 10

    def __getitem__(self, idx):
        idx = idx * 10
        return super().__getitem__(idx)
