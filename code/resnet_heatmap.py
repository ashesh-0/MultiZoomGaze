"""
Here, the code to get the heatmap of individual channels is present.
It assumes that a pretrained model of type GazeStaticSineAndCosineModel is passed on.
One has the option to choose the layer of which the heatmap is desired.
"""
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL import Image
from tqdm import tqdm

from eye_model.data_loader_static_sinecosine import DictEyeImgLoader, remove_eyeless_imgs
from backbones.resnet import ResNet
from sinecosine_model.data_loader_static_sinecosine import ImageLoaderStaticSineCosine
from sinecosine_model.static_sinecosine_model import GazeStaticSineAndCosineModel


class ResNetHeatmap(nn.Module):
    def __init__(self, model, idx):
        super().__init__()
        self._idx = idx
        layers = list(model.children())
        assert isinstance(layers[0], GazeStaticSineAndCosineModel)
        layers = list(layers[0].children())
        assert isinstance(layers[0], ResNet)
        resnet_layers = list(layers[0].children())
        print('-->'.join(f'{i}.{type(x).__name__}' for i, x in enumerate(resnet_layers)))
        self.fmap = nn.Sequential(*resnet_layers[:idx])

    def forward(self, input):
        return self.fmap(input)


class ImageLoaderHeatMap(ImageLoaderStaticSineCosine):
    def __init__(
            self,
            source_path,
            file_name=None,
            transform=None,
            g_z_min_val=None,
    ):
        super().__init__(source_path, file_name=file_name, transform=transform)
        loader_obj = DictEyeImgLoader()
        self.imgs = remove_eyeless_imgs(self.imgs, loader_obj)
        self._dict_loader = loader_obj.load

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        path_source, _ = self.imgs[index]
        return (img, self._dict_loader(path_source))

    def __len__(self):
        return len(self.imgs)


def get_blended_img(img: Image.Image, fmap: torch.Tensor, alpha: float) -> Image.Image:
    """
    Blends 2 images: img and fmap. Here fmap is actually the feature map. It is scaled up to img shape and
    with transparency `alpha`, the two images are superimposed.
    """
    h_img = F.to_pil_image(torch.Tensor(fmap.cpu().numpy()))
    hw = max(h_img.size[0], img.size[0])
    img = img.resize((hw, hw))
    h_img = h_img.resize((hw, hw))
    return Image.blend(img, h_img, alpha)


def blend_heatmap(img: Image.Image,
                  fmaps: np.ndarray,
                  blend_alpha: float = 0.8,
                  nrows: int = 8,
                  ncols: int = 8,
                  img_idx_list: List[int] = None):
    """
    This function plots multiple heatmaps either choosen at random if img_idx_list is None. Heatmaps are blended with
    the original image first before plotting. If img_idx_list is provided, only those channel heatmaps are plotted.
    Args:
        fmaps: Format: (#Channels, height, width)
    """
    if img_idx_list is None:
        cnt = nrows * ncols
        img_idx_list = np.random.choice(np.arange(fmaps.shape[0]), size=cnt, replace=False)
    else:
        cnt = len(img_idx_list)
        ncols = 8
        nrows = int(np.ceil(cnt / ncols))
    _, ax = plt.subplots(figsize=(20, 3 * nrows), nrows=nrows, ncols=ncols)

    for i, img_idx in enumerate(img_idx_list):
        bl_img = get_blended_img(img, fmaps[img_idx], blend_alpha)
        one_ax = ax[i // ncols, i % ncols] if nrows > 1 else ax[i]
        one_ax.imshow(bl_img)
        one_ax.set_title(f'#Channel:{img_idx}')


def get_eye_region_activation(activation: np.ndarray, eye_bbox: np.ndarray) -> np.ndarray:
    """
    This function computes the average activation in a rectangular region specified by eye_bbox. It is computed for all
    channels.
    Args:
        activation: Format: (#Channels, height, width)
        eye_bbox: array of size 4. They contain normalized x,y,h and w.

    """
    assert activation.shape[1] == activation.shape[2]
    y, x, h, w = (activation.shape[1] * eye_bbox).astype(int)
    return np.mean(activation[:, x:x + w, y:y + h].reshape((activation.shape[0], -1)), axis=1)


def get_eye_region_activation_df(model: ResNetHeatmap,
                                 img_loader: ImageLoaderHeatMap) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each image present in img_loader, the function computes the average activation of region of left and right eye.
    It computes it for each channel. It then computes several quantiles over the channel dimension.
    """
    quantiles = [0.5, 0.8, 0.9, 1.0]
    ldata = []
    rdata = []
    with torch.no_grad():
        for i in tqdm(range(len(img_loader))):
            img, img_dict = img_loader[i]
            activation = model(img.view((1, 3, 224, 224)))[0]
            l_activation = get_eye_region_activation(activation.cpu().numpy(), img_dict['bbox']['left'])
            r_activation = get_eye_region_activation(activation.cpu().numpy(), img_dict['bbox']['right'])
            ldata.append(np.quantile(l_activation, quantiles))
            rdata.append(np.quantile(r_activation, quantiles))
    leye_df = pd.DataFrame(ldata, columns=[f'Q{c}' for c in quantiles])
    reye_df = pd.DataFrame(ldata, columns=[f'Q{c}' for c in quantiles])
    return (leye_df, reye_df)
