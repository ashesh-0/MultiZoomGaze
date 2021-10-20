from typing import List

import numpy as np
import torchvision.transforms as transforms
from PIL.ImageOps import expand

from data_loader import default_loader


def custom_loader(crop_size):
    """
    Loads an image and either pads it or centercrops it.
    If crop_size > 224, it pads it to make it of size crop_size and then resize it to 224.
    If crop_size < 224, it centercrops it to size crop_size and then resize it to 224.
    """

    tr = transforms.Compose([
        transforms.Resize((224, 224)),
        CenterCropTransform([crop_size]),
    ], )

    def loader(path):
        img = default_loader(path)
        return tr(img)

    return loader


class CenterPadTransform:
    def __init__(self, pad_size):
        self._size = pad_size

    def __call__(self, img):
        return expand(img, self._size)


class CenterCropTransform:
    def __init__(self, sizes: List[int], output_size=224, pad=False, verbose=True):
        self._sizes = sizes
        self._output_size = output_size
        self._transforms = []
        self._pad = pad
        for size in self._sizes:
            if size <= self._output_size:
                if self._pad:
                    diff = (output_size - size)
                    val = diff // 2
                    # import pdb
                    # pdb.set_trace()
                    self._transforms.append(
                        transforms.Compose(
                            [transforms.CenterCrop(size),
                             transforms.Pad((val, val, diff - val, diff - val))]))
                else:
                    self._transforms.append(
                        transforms.Compose([transforms.CenterCrop(size),
                                            transforms.Resize(output_size)]))
            else:
                self._transforms.append(
                    transforms.Compose([CenterPadTransform(size - self._output_size),
                                        transforms.Resize(output_size)]))
        if verbose:
            print(f'[{self.__class__.__name__}] sizes:{self._sizes} pad:{self._pad} {self._transforms}')

    def __call__(self, img):
        i = np.random.choice(np.arange(len(self._transforms)))
        return self._transforms[i](img)
