from typing import List

import torch
import torchvision.transforms as transforms
from PIL import Image

from xgaze.xgaze_static_dataloader import GazeDataset


def centercrop_transform(crop_size, img_size, img_transform):
    return transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        img_transform,
    ])


class GazeMulticropDataset(GazeDataset):
    def __init__(self,
                 dataset_path: str,
                 keys_to_use: List[str] = None,
                 cropsizes=None,
                 sub_folder='',
                 transform=None,
                 is_shuffle=True,
                 index_file=None,
                 is_load_label=True):
        super().__init__(dataset_path,
                         keys_to_use=keys_to_use,
                         sub_folder=sub_folder,
                         transform=transform,
                         is_shuffle=is_shuffle,
                         index_file=index_file,
                         is_load_label=is_load_label)
        self._cropsizes = cropsizes
        self._transforms = [centercrop_transform(c, 224, transform) for c in self._cropsizes]
        print(f'[{self.__class__.__name__}] Cropsizes:{self._cropsizes}')

    def _get_img(self, hdf, hdf_idx):
        # Get face image
        image = hdf['face_patch'][hdf_idx, :]
        image = image[:, :, [2, 1, 0]]  # from BGR to RGB
        image = Image.fromarray(image)
        #
        seq_len = len(self._cropsizes)
        source_video = torch.FloatTensor(seq_len * 3, 224, 224)
        for i in range(seq_len):
            source_video[i * 3:i * 3 + 3, ...] = self._transforms[i](image)

        return {'raw': None, 'img': source_video}
