import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm

from data_loader import default_loader
from PIL import Image, ImageChops

MPIIFacegaze_BBOX = {}


def trim_loader(path_source):
    """
    Loads the image and trims it to clip the boundary regions. Needed for MPIIFaceGaze dataset.
    https://mail.python.org/pipermail/image-sig/2008-July/005092.html
    """
    im = default_loader(path_source)
    if path_source not in MPIIFacegaze_BBOX:
        bg = Image.new(im.mode, im.size, 'black')
        diff = ImageChops.difference(im, bg)
        bbox = diff.getbbox()
        if bbox:
            MPIIFacegaze_BBOX[path_source] = bbox
        else:
            # found no content
            raise ValueError(f"cannot trim; image was empty {path_source}")
    else:
        bbox = MPIIFacegaze_BBOX[path_source]

    return im.crop(bbox)


# gt => gaze target.
# fc => face center. approximated as eye center.
# fl => facial landmarks.
# hp => head pose.
# g_2D => 2D gaze.
MPIIGazeFaceMetadataColumns = ['path', 'g_2Dx', 'g_2D_y'] + [f'fl_{i}' for i in range(12)] + [
    f'hp_{i}' for i in range(6)
] + ['fc_x', 'fc_y', 'fc_z'] + ['gt_x', 'gt_y', 'gt_z'] + ['eye']


def make_dataset_MPIIGazeFace(data_dir):
    images = []

    for person in tqdm(os.listdir(data_dir)):
        if not os.path.isdir(os.path.join(data_dir, person)):
            continue
        meta_fname = os.path.join(data_dir, person, f'{person}.txt')
        metadata_df = pd.read_csv(meta_fname, names=MPIIGazeFaceMetadataColumns, delimiter=' ')
        for _, row in metadata_df.iterrows():
            img_fpath = os.path.join(data_dir, person, row['path'])
            gaze = row[['gt_x', 'gt_y', 'gt_z']].values - row[['fc_x', 'fc_y', 'fc_z']].values
            gaze = gaze / np.linalg.norm(gaze)
            # change the co-ordinate so as to align with Gaze360.
            gaze[1] = -1 * gaze[1]
            gaze[0] = -1 * gaze[0]
            images.append((img_fpath, gaze.astype(np.float32)))
    return images


def make_dataset_gaze360(source_path, file_name, g_z_min_val=None):
    """
    If g_z_min_val is given, then skip entries with g_z less than g_z_min_val.
    """
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
                gaze = np.zeros((3))
                gaze[0] = float(split_lines[1])
                gaze[1] = float(split_lines[2])
                gaze[2] = float(split_lines[3])
                if g_z_min_val is not None and gaze[2] < g_z_min_val:
                    skip_count += 1
                    continue

                item = (os.path.join(source_path, split_lines[0]), gaze)
                images.append(item)
    if skip_count > 0:
        print('Skipped', skip_count / (skip_count + len(images)))
    return images


class DatasetType:
    gaze360 = 'Gaze360'
    mpiifacegaze = 'MPIIFaceGaze'


class ImagerLoaderStaticModel(data.Dataset):
    def __init__(
        self,
        source_path,
        file_name=None,
        transform=None,
        target_transform=None,
        loader=default_loader,
        dataset_type=DatasetType.gaze360,
    ):
        if dataset_type == DatasetType.gaze360:
            imgs = make_dataset_gaze360(source_path, file_name)
        elif dataset_type == DatasetType.mpiifacegaze:
            imgs = make_dataset_MPIIGazeFace(source_path)
        else:
            raise Exception('Invalid dataset_type:', dataset_type)

        self.source_path = source_path
        self.file_name = file_name

        self.imgs = imgs
        self.transform = transform
        self.target_transform = transform
        self.loader = loader

    def get_img(self, index):
        path_source, _ = self.imgs[index]
        return torch.FloatTensor(self.transform(self.loader(path_source)))

    def get_xyz_gaze(self, index):
        _, gaze = self.imgs[index]

        gaze_float = torch.Tensor(gaze)
        gaze_float = torch.FloatTensor(gaze_float)
        normalized_gaze = nn.functional.normalize(gaze_float.view(1, 3)).view(3)
        return normalized_gaze

    def get_yawpitch_gaze(self, index):
        normalized_gaze = self.get_xyz_gaze(index)
        spherical_vector = torch.FloatTensor(2)
        # yaw
        spherical_vector[0] = math.atan2(normalized_gaze[0], -normalized_gaze[2])
        # pitch
        spherical_vector[1] = math.asin(normalized_gaze[1])
        return spherical_vector

    def get_gaze(self, index):
        return self.get_yawpitch_gaze(index)

    def __getitem__(self, index):
        source_img = self.get_img(index)
        spherical_vector = self.get_yawpitch_gaze(index)
        return source_img, spherical_vector

    def __len__(self):
        return len(self.imgs)
