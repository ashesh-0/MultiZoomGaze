import numpy as np


def get_orig_img(img):
    orig_img = (img - img.min().item())
    orig_img = orig_img[0]
    orig_img = 255 * orig_img / (orig_img.max().item())
    orig_img = np.swapaxes(np.swapaxes(orig_img.detach().numpy(), 0, 1), 1, 2).astype(int)
    return orig_img
