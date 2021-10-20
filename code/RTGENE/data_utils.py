import os
import re

import numpy as np

from core.enum import Enum


class DataType(Enum):
    Inpainted = 0
    InpaintedLarge = 1
    Original = 2
    OriginalLarge = 3


def create_label_dict(label_fpath):
    """
    Example line: "4, [0.0439945012147, -0.420823062146], [-0.121331775145, -0.218748397287], 1504604977.99"
    """
    output = {}
    with open(label_fpath, 'r') as f:
        lines = f.read().splitlines()
    for line in lines:
        tokens = ' '.join(re.split(',|\[|\]', line)).split()
        assert len(tokens) == 6
        # Yaw
        gaze_phi = float(tokens[3])
        # Pitch
        gaze_theta = float(tokens[4])
        idx = int(tokens[0])
        assert idx not in output
        output[idx] = np.array([gaze_phi, gaze_theta], dtype=np.float32)
    return output


def fname_from_id(idx, data_type):
    if data_type in [DataType.OriginalLarge, DataType.Original, DataType.Inpainted]:
        return f'face_{idx:06d}_rgb.png'
    return None


def data_subfolder(data_type):
    if data_type == DataType.OriginalLarge:
        return 'original/face_before_inpainting/'
    elif data_type == DataType.Original:
        return 'original/face/'
    elif data_type == DataType.Inpainted:
        return 'inpainted/face'

    return None


def create_dataset(directory, person_list, data_type=DataType.OriginalLarge):
    dataset = []
    assert data_type in [DataType.OriginalLarge, DataType.Inpainted, DataType.Original], DataType.name(data_type)
    for person in person_list:
        # img_folder = os.path.join(directory, person, 'inpainted/face')
        img_folder = os.path.join(directory, person, data_subfolder(data_type))
        label_fpath = os.path.join(directory, person, 'label_combined.txt')
        label_dict = create_label_dict(label_fpath)
        for idx, target in label_dict.items():
            fname = fname_from_id(idx, data_type)
            entry = (os.path.join(img_folder, fname), target)
            dataset.append(entry)
    print(f'create_dataset {DataType.name(data_type)}: {len(dataset)/1000}K sized dataset created.')
    return dataset
