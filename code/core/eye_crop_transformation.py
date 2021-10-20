from typing import Dict, Union

import numpy as np
from PIL import Image

from core.dict_img_loader import scale_bbox


class EyeCropTransformation:
    def __init__(self, magnify_eye_region_factor=1.5):
        self._magnify_eye_region_factor = magnify_eye_region_factor
        print(f'[{self.__class__.__name__}] Fac:{self._magnify_eye_region_factor}')

    def get_pixel_coordinates(self, input_, img_shape):
        bbox = input_['bbox']
        assert len(img_shape) == 3 and abs(img_shape[0] - img_shape[1]) <= 2, img_shape

        leye_b = scale_bbox(bbox['left'], self._magnify_eye_region_factor)
        reye_b = scale_bbox(bbox['right'], self._magnify_eye_region_factor)
        # print(leye_b, reye_b)
        leye_b = (leye_b * img_shape[0]).astype(int)
        reye_b = (reye_b * img_shape[0]).astype(int)

        assert set(leye_b) != {-img_shape[0]}
        assert set(reye_b) != {-img_shape[0]}
        return leye_b, reye_b

    def __call__(self, input_: Dict[str, Union[np.array, Image.Image]]) -> Image.Image:
        """
        Args:
            input_['bbox']['left']: 4 valued array: x,y,w,h. input_[x:x+w,y:y+h] region is returned. For left eye.
            input_['bbox']['right']: Same as 'left'.For right eye.
        """
        img = np.array(input_['img'])
        leye_b, reye_b = self.get_pixel_coordinates(input_, img.shape)
        leye = img[leye_b[1]:leye_b[1] + leye_b[3], leye_b[0]:leye_b[0] + leye_b[2]]
        reye = img[reye_b[1]:reye_b[1] + reye_b[3], reye_b[0]:reye_b[0] + reye_b[2]]
        return Image.fromarray(leye), Image.fromarray(reye)


class EyeRegionCropTransformation(EyeCropTransformation):
    def __call__(self, input_: Dict[str, Union[np.array, Image.Image]]) -> Image.Image:
        """
        Args:
            input_['bbox']['left']: 4 valued array: x,y,w,h. input_[x:x+w,y:y+h] region is returned. For left eye.
            input_['bbox']['right']: Same as 'left'.For right eye.
        """
        img = np.array(input_['img'])
        leye_b, reye_b = self.get_pixel_coordinates(input_, img.shape)
        # y_min = leye_b[1]
        # y_max = reye_b[1] + reye_b[3]

        y_min = min(leye_b[1], reye_b[1])
        y_max = max(leye_b[1] + leye_b[3], reye_b[1] + reye_b[3])

        x_min = min(leye_b[0], reye_b[0])
        x_max = max(leye_b[0] + leye_b[2], reye_b[0] + reye_b[2])
        eye_region = img[y_min:y_max, x_min:x_max]
        return Image.fromarray(eye_region)


class FixedSizeCropTransformation(EyeCropTransformation):
    def __init__(self, magnify_eye_region_factor=1.5, img_position='H'):
        super().__init__(magnify_eye_region_factor=magnify_eye_region_factor)
        assert img_position in ['H', 'V', None]
        self._img_position = img_position

    @staticmethod
    def get_size_factors(img_position):
        if img_position == 'H':
            Ysize_factor = 1
            Xsize_factor = 0.5
        elif img_position == 'V':
            Ysize_factor = 0.5
            Xsize_factor = 1
        return Xsize_factor, Ysize_factor

    @staticmethod
    def expand(start_end, max_val, sz):
        """
        Interval [s,e] needs to get expanded to [s1,e1] such that e1-s1 == sz and e1 =< max_val
        """
        s, e = start_end
        # print(s, e)
        assert s >= 0 and e >= 0 and isinstance(s, int) and isinstance(e, int)

        diff = sz - (e - s)
        if diff <= 0:
            return (s, e)

        if sz >= max_val:
            return (0, max_val)

        s -= int(np.ceil(diff / 2))
        e += int(np.floor(diff / 2))

        if s < 0:
            e += (-s)
            s = 0
        if e >= max_val:
            s -= (e - max_val + 1)
            e = max_val - 1

        assert e - s == sz
        return (s, e)

    def get_combined_eyes_pixel_coordinates(self, input_, img_shape=None):
        if img_shape is None:
            img_shape = (*input_['img'].size, 3)
        leye_b, reye_b = self.get_pixel_coordinates(input_, img_shape)

        y_min = min(leye_b[1], reye_b[1])
        y_max = max(leye_b[1] + leye_b[3], reye_b[1] + reye_b[3])

        x_min = min(leye_b[0], reye_b[0])
        x_max = max(leye_b[0] + leye_b[2], reye_b[0] + reye_b[2])

        x_min = int(x_min)
        x_max = int(x_max)
        y_min = int(y_min)
        y_max = int(y_max)
        assert x_min >= 0 and y_min >= 0
        return (x_min, x_max, y_min, y_max)

    def get_eye_region_pixel_coordinates(self, input_, img_position: str = None):
        img = np.array(input_['img'])
        (x_min, x_max, y_min, y_max) = self.get_combined_eyes_pixel_coordinates(input_)
        assert img.shape[2] == 3, img.shape
        Y_max, X_max = img.shape[:2]
        if img_position is None:
            assert self._img_position is not None
            img_position = self._img_position
        else:
            self._img_position is None
        Xsize_factor, Ysize_factor = self.get_size_factors(img_position)

        y_min, y_max = self.expand((y_min, y_max), Y_max, int(Y_max * Ysize_factor))
        x_min, x_max = self.expand((x_min, x_max), X_max, int(X_max * Xsize_factor))
        return (x_min, x_max, y_min, y_max)

    def __call__(self, input_: Dict[str, Union[np.array, Image.Image]], img_position: str = None) -> Image.Image:
        """
        Args:
            input_['bbox']['left']: 4 valued array: x,y,w,h. input_[x:x+w,y:y+h] region is returned. For left eye.
            input_['bbox']['right']: Same as 'left'.For right eye.
        """
        img = np.array(input_['img'])
        (x_min, x_max, y_min, y_max) = self.get_eye_region_pixel_coordinates(input_, img_position=img_position)
        img_region = img[y_min:y_max, x_min:x_max]
        return Image.fromarray(img_region)
