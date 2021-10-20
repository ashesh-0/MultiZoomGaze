import math

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from sinecosine_model.data_loader_sinecosine import ImageLoaderSineCosine


class RandomZoomImageLoader(ImageLoaderSineCosine):
    def __init__(self, center_crop_sz, source_path, file_name, transform=None):
        super().__init__(source_path, file_name, transform=transform)
        self._crop = center_crop_sz
        image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform2 = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(self._crop),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            image_normalize,
        ])

    def _get_sinecosine_gaze(self, gaze_xyz):
        gaze_float = torch.Tensor(gaze_xyz)
        gaze_float = torch.FloatTensor(gaze_float)
        normalized_gaze = nn.functional.normalize(gaze_float.view(1, 3)).view(3)
        spherical_vector = torch.FloatTensor(2)
        # yaw
        spherical_vector[0] = math.atan2(normalized_gaze[0], -normalized_gaze[2])
        # pitch
        spherical_vector[1] = math.asin(normalized_gaze[1])

        gaze_sincos_vector = torch.Tensor([0., 0., 0.])
        #sin(Yaw)
        gaze_sincos_vector[0] = torch.sin(spherical_vector[0])
        # cos(Yaw)
        gaze_sincos_vector[1] = torch.cos(spherical_vector[0])
        # sin(Pitch)
        gaze_sincos_vector[2] = torch.sin(spherical_vector[1])
        return gaze_sincos_vector

    def _get_video(self, path_source):
        source_video = torch.FloatTensor(7, 3, 224, 224)
        reverse_it = self._reverse and np.random.rand() > 0.5

        if self._random_walk:
            path_source = self.get_random_walk_paths(path_source, self._rw_forward_bias)

        for i, frame_path in enumerate(path_source):
            transform = self.transform if np.random.rand() > 0.5 else self.transform2
            if reverse_it:
                source_video[6 - i, ...] = transform(self.loader(frame_path))
            else:
                source_video[i, ...] = transform(self.loader(frame_path))

        source_video = source_video.view(21, 224, 224)
        return source_video

    def __getitem__(self, index):
        path_source, gaze = self.imgs[index]
        return self._get_video(path_source), self._get_sinecosine_gaze(gaze)


class ZoomKthImageLoader(RandomZoomImageLoader):
    def __init__(self, k, center_crop_sz, source_path, file_name, transform=None):
        super().__init__(center_crop_sz, source_path, file_name, transform=transform)
        self._k = k
        assert isinstance(self._k, int)
        assert self._k > 0 and self._k <= 7
        print(f'[{self.__class__.__name__}] centercrop:{center_crop_sz} K:{k}')

    def _get_video(self, path_source):
        source_video = torch.FloatTensor(7, 3, 224, 224)
        for i, frame_path in enumerate(path_source):
            transform = self.transform2 if i == self._k - 1 else self.transform
            source_video[i, ...] = transform(self.loader(frame_path))
        source_video = source_video.view(21, 224, 224)
        return source_video
