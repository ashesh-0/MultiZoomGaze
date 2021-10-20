import math
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg',
    '.JPG',
    '.jpeg',
    '.JPEG',
    '.png',
    '.PNG',
    '.ppm',
    '.PPM',
    '.bmp',
    '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset_future_prediction(source_path, file_name, g_z_min_val=None, enable_time=True, seq_len=7):
    """
    Here, we predict the gaze for the last frame. Usually (in make_dataset()), we predict for middle frame.
    """
    assert enable_time is True

    assert seq_len > 0
    assert seq_len % 2 == 1 or enable_time is False
    images = []
    print(file_name)
    skip_count = 0
    with open(file_name, 'r') as f:
        for line in f:
            line = line[:-1]
            line = line.replace("\t", " ")
            line = line.replace("  ", " ")
            split_lines = line.split(" ")
            if (len(split_lines) > 3):
                frame_number = int(split_lines[0].split('/')[-1][:-4])
                lists_sources = []
                # import pdb
                # pdb.set_trace()
                for j in range(-seq_len + 1, 1):
                    new_frame_number = int(frame_number + j * int(enable_time))
                    name_frame = '/'.join(split_lines[0].split('/')[:-1] + ['%0.6d.jpg' % (new_frame_number)])
                    name = '{0}/{1}'.format(source_path, name_frame)
                    lists_sources.append(name)

                gaze = np.zeros((3))

                gaze[0] = float(split_lines[1])
                gaze[1] = float(split_lines[2])
                gaze[2] = float(split_lines[3])
                if g_z_min_val is not None and gaze[2] < g_z_min_val:
                    skip_count += 1
                    continue
                if not all([os.path.exists(nm) for nm in lists_sources]):
                    skip_count += 1
                    continue

                item = (lists_sources, gaze)
                images.append(item)

    if skip_count > 0:
        print('Skipped', skip_count / (skip_count + len(images)))

    return images


def make_dataset(source_path, file_name, g_z_min_val=None, enable_time=True, seq_len=7):
    """
    If g_z_min_val is given, then skip entries with g_z less than g_z_min_val.
    """
    assert seq_len > 0
    assert seq_len % 2 == 1 or enable_time is False
    images = []
    print(file_name)
    skip_count = 0
    with open(file_name, 'r') as f:
        for line in f:
            line = line[:-1]
            line = line.replace("\t", " ")
            line = line.replace("  ", " ")
            split_lines = line.split(" ")
            if (len(split_lines) > 3):
                frame_number = int(split_lines[0].split('/')[-1][:-4])
                lists_sources = []
                # import pdb
                # pdb.set_trace()
                for j in range(-(seq_len // 2), seq_len // 2 + seq_len % 2):
                    new_frame_number = int(frame_number + j * int(enable_time))
                    name_frame = '/'.join(split_lines[0].split('/')[:-1] + ['%0.6d.jpg' % (new_frame_number)])
                    name = '{0}/{1}'.format(source_path, name_frame)
                    lists_sources.append(name)

                gaze = np.zeros((3))

                gaze[0] = float(split_lines[1])
                gaze[1] = float(split_lines[2])
                gaze[2] = float(split_lines[3])
                if g_z_min_val is not None and gaze[2] < g_z_min_val:
                    skip_count += 1
                    continue

                item = (lists_sources, gaze)
                images.append(item)

    if skip_count > 0:
        print('Skipped', skip_count / (skip_count + len(images)))

    return images


def default_loader(path):
    try:
        im = Image.open(path).convert('RGB')
        return im
    except OSError:
        print(path)
        return Image.new("RGB", (512, 512), "white")


class ImagerLoader(data.Dataset):
    def __init__(
        self,
        source_path,
        file_name,
        transform=None,
        loader=default_loader,
        g_z_min_val=None,
        reverse=False,
        random_walk=False,
        rw_forward_bias=0.7,
        enable_time=True,
        future_prediction=False,
        seq_len=7,
        img_size=224,
    ):

        self._enable_time = enable_time
        self.source_path = source_path
        self.file_name = file_name
        self.transform = transform
        self.loader = loader
        self._reverse = reverse
        self._random_walk = random_walk
        self._rw_forward_bias = rw_forward_bias
        self._seq_len = seq_len
        self._img_size = img_size
        if future_prediction:
            imgs = make_dataset_future_prediction(source_path,
                                                  file_name,
                                                  g_z_min_val=g_z_min_val,
                                                  enable_time=self._enable_time,
                                                  seq_len=seq_len)

        else:
            imgs = make_dataset(source_path,
                                file_name,
                                g_z_min_val=g_z_min_val,
                                enable_time=self._enable_time,
                                seq_len=seq_len)
        self.imgs = imgs
        print(f'[ImageLoader] reverse:{self._reverse} random_walk:{self._random_walk} forward_bias:'
              f'{self._rw_forward_bias} Time:{self._enable_time} SeqLen:{seq_len} ImgSize:{img_size} '
              f'FP:{future_prediction}')

    def _get_yaw_pitch(self, gaze_xyz):
        gaze_float = torch.Tensor(gaze_xyz)
        gaze_float = torch.FloatTensor(gaze_float)
        normalized_gaze = nn.functional.normalize(gaze_float.view(1, 3)).view(3)
        spherical_vector = torch.FloatTensor(2)
        # yaw
        spherical_vector[0] = math.atan2(normalized_gaze[0], -normalized_gaze[2])
        # pitch
        spherical_vector[1] = math.asin(normalized_gaze[1])
        return spherical_vector

    def _get_video(self, path_source):
        source_video = torch.FloatTensor(self._seq_len, 3, self._img_size, self._img_size)
        reverse_it = self._reverse and np.random.rand() > 0.5

        if self._random_walk:
            path_source = self.get_random_walk_paths(path_source, self._rw_forward_bias)

        for i, frame_path in enumerate(path_source):
            if reverse_it:
                source_video[self._seq_len - 1 - i, ...] = self.transform(self.loader(frame_path))
            else:
                source_video[i, ...] = self.transform(self.loader(frame_path))

        source_video = source_video.view(self._seq_len * 3, self._img_size, self._img_size)
        return source_video

    def __getitem__(self, index):
        path_source, gaze = self.imgs[index]
        source_video = self._get_video(path_source)
        return source_video, self._get_yaw_pitch(gaze)

    def __len__(self):
        return len(self.imgs)

    def get_random_walk_paths(self, paths: List[str], forward_bias: float):
        N = len(paths)
        assert N % 2 == 1
        mid = N // 2
        left_portion = self._get_random_walk(mid, N // 2 + 1, 1 - forward_bias)
        right_portion = self._get_random_walk(mid, N // 2 + 1, forward_bias)

        img_sequence = left_portion[::-1] + right_portion[1:]
        return [paths[i] for i in img_sequence]

    def _get_random_walk(self, start: int, steps: int, forward_bias: float):
        walk = [start]
        for _ in range(steps - 1):
            current = walk[-1]
            p = np.random.rand()
            if p <= forward_bias:
                walk.append(current + 1)
            else:
                walk.append(current - 1)
        return walk
