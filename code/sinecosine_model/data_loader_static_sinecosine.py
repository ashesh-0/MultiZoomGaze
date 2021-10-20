"""
ImageLoader classes for multiple variants of sin/cosine models where we return sine/cosine of yaw and pitch instead of
predicting actual yaw and pitch. In some cases, a boolean vector is also returned additionally.
"""
from typing import List

import numpy as np
import torch

from core.analysis_utils import get_frame, get_person, get_session
from core.extended_head import ExtendedHead
from data_loader_static_model import ImagerLoaderStaticModel
from sinecosine_model.train_utils import get_multi_sincos_gaze_from_spherical, get_sincos_gaze_from_spherical


def crop_img(img, new_x, new_y):
    x, y = img.size
    left = (x - new_x) / 2
    top = (y - new_y) / 2
    right = (x + new_x) / 2
    bottom = (y + new_y) / 2

    # Crop the center of the image
    img = img.crop((left, top, right, bottom))
    return img


class ImageLoaderStaticSineCosine(ImagerLoaderStaticModel):
    """
    ImageLoader of Static model. Here, we return sin(yaw),cos(yaw), sin(pitch)
    """

    def __getitem__(self, index):
        source_img, spherical_vector = super().__getitem__(index)
        gaze_sine_vector = get_sincos_gaze_from_spherical(spherical_vector)
        return source_img, gaze_sine_vector


class ImageLoaderStaticMultiSineCosine(ImagerLoaderStaticModel):
    """
    ImageLoader of Static model. Here, we return sin(yaw),cos(yaw), sin(pitch)
    """

    def __getitem__(self, index):
        source_img, spherical_vector = super().__getitem__(index)
        gaze_sine_vector = get_multi_sincos_gaze_from_spherical(spherical_vector)
        return source_img, gaze_sine_vector


class ImageLoaderStaticSineCosineMultiCenterCrops(ImageLoaderStaticSineCosine):
    def __init__(
            self,
            source_path,
            file_name,
            transform,
            cropsize_list: List[int] = None,
            finegrained_randomness=False,
    ):
        super().__init__(source_path, file_name=file_name, transform=transform)
        self._c_list = cropsize_list
        self._finegrained_randomness = finegrained_randomness
        self._large_head_fetcher = ExtendedHead(file_name, skip_improper_imgs=False)
        self._input_size = 224
        print(f'[{self.__class__.__name__}] Cropsize list:{self._c_list} '
              f'FinegrainedRandom:{self._finegrained_randomness}')

    def get_img(self, index):
        path_source, _ = self.imgs[index]
        if self._finegrained_randomness:
            csize = np.random.choice(np.arange(min(self._c_list), max(self._c_list) + 1))
        else:
            csize = np.random.choice(self._c_list)
        # print(csize)
        if csize <= self._input_size:
            img = self.loader(path_source).resize((self._input_size, self._input_size))
            img = crop_img(img, csize, csize)
        else:

            m = 224 / csize
            session = get_session(path_source)
            person = get_person(path_source)
            frame = get_frame(path_source)
            img = self._large_head_fetcher.get(session, person, frame, 0, 0, m)['img']
        return torch.FloatTensor(self.transform(img))
