import os

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from data_loader import default_loader
from data_loader_static_model import make_dataset_gaze360


class ImagerLoaderStaticXyzModel(data.Dataset):
    def __init__(
            self,
            source_path,
            file_name=None,
            transform=None,
            target_transform=None,
            loader=default_loader,
    ):
        imgs = make_dataset_gaze360(source_path, file_name)

        self.source_path = source_path
        self.file_name = file_name

        self.imgs = imgs
        self.transform = transform
        self.target_transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path_source, gaze = self.imgs[index]

        gaze_float = torch.Tensor(gaze)
        gaze_float = torch.FloatTensor(gaze_float)
        normalized_gaze = nn.functional.normalize(gaze_float.view(1, 3)).view(3)

        source_img = torch.FloatTensor(3, 224, 224)
        source_img[:] = self.transform(self.loader(path_source))

        return source_img, normalized_gaze

    def __len__(self):
        return len(self.imgs)
