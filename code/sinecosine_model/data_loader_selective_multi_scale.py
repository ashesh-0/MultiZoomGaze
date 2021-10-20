from typing import List

import torch

from sinecosine_model.data_loader_sinecosine import ImageLoaderSineCosine, centercrop_transform


class ImageLoaderSineCosineSelectiveMultiScale(ImageLoaderSineCosine):
    """
    Here, we apply multi central crops to central image. For other frames we have a single cropsize.
    """

    def __init__(
            self,
            source_path,
            file_name,
            transform=None,
            cropsize_list: List[int] = None,
            central_cropsize_list: List[int] = None,
            reverse=False,
            enable_time=True,
            seq_len=7,
            img_size=224,
    ):
        super().__init__(
            source_path,
            file_name,
            transform=None,
            reverse=reverse,
            enable_time=enable_time,
            seq_len=seq_len,
            img_size=img_size,
        )
        assert cropsize_list is not None
        assert central_cropsize_list is not None
        self._cropsizes = cropsize_list
        assert len(self._cropsizes) == self._seq_len - 1

        self._central_cropsizes = central_cropsize_list
        self._transforms = [centercrop_transform(c, img_size) for c in self._cropsizes]
        self._central_transforms = [centercrop_transform(c, img_size) for c in self._central_cropsizes]
        print(f'[{self.__class__.__name__}] Cropsizes:{self._cropsizes} CentralCropsize:{self._central_cropsizes}')

    def _get_central_frames(self, path):
        img = self.loader(path)
        central_frame = torch.FloatTensor(len(self._central_cropsizes), 3, self._img_size, self._img_size)
        for i in range(len(self._central_cropsizes)):
            central_frame[i, ...] = self._central_transforms[i](img)
        return central_frame

    def _get_video(self, path_source):
        assert len(path_source) == self._seq_len
        assert self._reverse is False
        mid_path = path_source[len(path_source) // 2]
        central_frames = self._get_central_frames(mid_path)
        # skip the middle fpath
        path_source = [fpath for i, fpath in enumerate(path_source) if i != len(path_source) // 2]

        source_video = torch.FloatTensor(self._seq_len - 1, 3, self._img_size, self._img_size)
        # central_frame = torch.FloatTensor(len(self._central_cropsizes), 3, self._img_size, self._img_size)
        for i, frame_path in enumerate(path_source):
            source_video[i, ...] = self._transforms[i](self.loader(frame_path))

        # source_video = source_video.view(3 * (self._seq_len - 1), self._img_size, self._img_size)
        return source_video, central_frames
