from typing import List

import torch

from core.transforms import centercrop_transform
from data_loader import ImagerLoader, default_loader


class ImageLoaderMultiSizedCrops(ImagerLoader):
    def __init__(
            self,
            source_path,
            file_name,
            transform=None,
            cropsize_list: List[int] = None,
            enable_time=True,
    ):
        super().__init__(
            source_path,
            file_name,
            transform=None,
            reverse=False,
            enable_time=enable_time,
            seq_len=len(cropsize_list),
            img_size=224,
            loader=default_loader,
        )
        self._cropsizes = cropsize_list
        self._transforms = [centercrop_transform(c, self._img_size) for c in self._cropsizes]
        print(f'[{self.__class__.__name__}] Cropsizes:{self._cropsizes}')

    def _get_video(self, path_source):
        source_video = torch.FloatTensor(self._seq_len, 3, self._img_size, self._img_size)
        for i, frame_path in enumerate(path_source):
            source_video[i, ...] = self._transforms[i](self.loader(frame_path))

        source_video = source_video.view(3 * self._seq_len, self._img_size, self._img_size)
        return source_video
