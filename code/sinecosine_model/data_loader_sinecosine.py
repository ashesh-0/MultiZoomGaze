from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from core.transforms import centercrop_transform
from data_loader import ImagerLoader, default_loader


class ImageLoaderSineCosine(ImagerLoader):
    """
    ImageLoader of LSTM model. Here, we return sin(yaw),cos(yaw) and sin(pitch)
    """
    def __getitem__(self, index):
        source_video, spherical_vector = super().__getitem__(index)
        gaze_sincos_vector = torch.Tensor([0., 0., 0.])
        #sin(Yaw)
        gaze_sincos_vector[0] = torch.sin(spherical_vector[0])
        # cos(Yaw)
        gaze_sincos_vector[1] = torch.cos(spherical_vector[0])
        # sin(Pitch)
        gaze_sincos_vector[2] = torch.sin(spherical_vector[1])

        return source_video, gaze_sincos_vector


class ImageLoaderSineCosineMultiSizedCrops(ImageLoaderSineCosine):
    def __init__(
        self,
        source_path,
        file_name,
        transform=None,
        cropsize_list: List[int] = None,
        reverse=False,
        enable_time=True,
        future_prediction=False,
        seq_len=7,
        img_size=224,
        loader=default_loader,
    ):
        super().__init__(
            source_path,
            file_name,
            transform=None,
            reverse=reverse,
            enable_time=enable_time,
            future_prediction=future_prediction,
            seq_len=seq_len,
            img_size=img_size,
            loader=loader,
        )
        self._cropsizes = cropsize_list
        self._transforms = [centercrop_transform(c, img_size) for c in self._cropsizes]
        print(f'[{self.__class__.__name__}] Cropsizes:{self._cropsizes}')

    def _get_video_with_transforms(self, path_source, transforms):
        # import pdb
        # pdb.set_trace()
        source_video = torch.FloatTensor(self._seq_len, 3, self._img_size, self._img_size)
        reverse_it = self._reverse and np.random.rand() > 0.5
        for i, frame_path in enumerate(path_source):
            if reverse_it:
                source_video[self._seq_len - 1 - i, ...] = transforms[i](self.loader(frame_path))
            else:
                source_video[i, ...] = transforms[i](self.loader(frame_path))

        source_video = source_video.view(3 * self._seq_len, self._img_size, self._img_size)
        return source_video

    def _get_video(self, path_source):
        return self._get_video_with_transforms(path_source, self._transforms)


class ImageLoaderSineCosineMultiSizedRandomCrops(ImageLoaderSineCosineMultiSizedCrops):
    def _get_video(self, path_source):
        order = np.random.permutation(np.arange(self._seq_len))
        transforms = [self._transforms[i] for i in order]
        return self._get_video_with_transforms(path_source, transforms)


def crop_tensor(variable, sz):
    _, w, h = variable.shape
    x1 = int(round((w - sz) / 2.))
    y1 = int(round((h - sz) / 2.))
    return variable[:, x1:x1 + sz, y1:y1 + sz]


def crop_tensorv2(variable, sz):
    _, _, w, h = variable.shape
    x1 = int(round((w - sz) / 2.))
    y1 = int(round((h - sz) / 2.))
    return variable[:, :, x1:x1 + sz, y1:y1 + sz]


class ImageLoaderSineCosineMultiScale(ImageLoaderSineCosine):
    def __init__(
        self,
        source_path,
        file_name,
        transform=None,
        cropsize_list: List[int] = None,
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
        assert reverse is False
        self._cropsizes = cropsize_list
        # self._transforms = [centercrop_transform(c, img_size) for c in self._cropsizes]
        image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            image_normalize,
        ])
        print(f'[{self.__class__.__name__}] Cropsizes:{self._cropsizes}')

    # def _get_video(self, path_source):
    #     source_video = torch.FloatTensor(self._seq_len, len(self._cropsizes), 3, self._img_size, self._img_size)
    #     reverse_it = self._reverse and np.random.rand() > 0.5

    #     for i, frame_path in enumerate(path_source):
    #         img = self._transform(self.loader(frame_path))
    #         for sc_idx in range(len(self._cropsizes)):
    #             sz = self._cropsizes[sc_idx]
    #             # import pdb
    #             # pdb.set_trace()
    #             if sz != img.shape[-1]:
    #                 cropped_img = crop_tensor(img, sz)
    #             else:
    #                 cropped_img = img

    #             inpt = F.interpolate(
    #                 cropped_img[None, ...], (self._img_size, self._img_size), mode='bilinear', align_corners=False)[0]

    #             if reverse_it:
    #                 source_video[self._seq_len - 1 - i, sc_idx, ...] = inpt
    #             else:
    #                 source_video[i, sc_idx, ...] = inpt

    #     source_video = source_video.view(3 * self._seq_len * len(self._cropsizes), self._img_size, self._img_size)
    #     return source_video

    def _get_video(self, path_source):

        frames = []
        for i, frame_path in enumerate(path_source):
            img = self._transform(self.loader(frame_path))
            frames.append(img[None, ...])

        # 7,3,224,224
        video = torch.cat(frames, dim=0)
        cropped_video_list = []
        for sc_idx in range(len(self._cropsizes)):
            sz = self._cropsizes[sc_idx]
            if sz != 224:
                cropped_video = crop_tensorv2(video, sz)
            else:
                cropped_video = video

            resized_video = F.interpolate(cropped_video, (self._img_size, self._img_size),
                                          mode='bilinear',
                                          align_corners=False)
            cropped_video_list.append(resized_video[:, None, ...])

        output_video = torch.cat(cropped_video_list, dim=1)
        output_video = output_video.view(3 * self._seq_len * len(self._cropsizes), self._img_size, self._img_size)
        return output_video
