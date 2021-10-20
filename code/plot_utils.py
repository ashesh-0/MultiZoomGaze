from typing import Tuple

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from core.analysis_utils import get_filename


def draw_gaze(image_in: np.ndarray,
              pos: Tuple[int, int],
              pitchyaw: Tuple[float, float],
              length=40.0,
              thickness=1,
              tiplength=0.2,
              color=(0, 0, 255)):
    """
    Draw gaze angle on given image with a given eye positions.
    Args:
        image_in: input image
        pos: position where the base of arrow needs to be placed.
        pitchyaw: A tuple containing [pitch angle, yaw angle]
        length: length of arrow

    Adapted from https://github.com/xucong-zhang/ETH-XGaze/blob/master/normalization_example.py
    """
    import cv2
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    else:
        image_out = cv2.UMat(image_out)

    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])
    # import pdb
    # pdb.set_trace()
    cv2.arrowedLine(image_out,
                    tuple(np.round(pos).astype(np.int32)),
                    tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)),
                    color,
                    thickness,
                    cv2.LINE_AA,
                    tipLength=tiplength)
    return cv2.UMat.get(image_out)


def draw_target_predicted_gaze(entry):
    """
     Given one entry of a dataframe containing session, person, frame, g_yaw, g_pitch, pred_yaw_weighted and
     pred_pitch_cos, get the image and draw correct and predicted gazes.
    """
    fpath = get_filename(entry['session'], entry['person'], entry['frame'], crop_type='head')
    img = np.array(Image.open(fpath))
    new_img = draw_gaze(img, (img.shape[0] // 2, img.shape[1] // 2), (entry['g_pitch'], entry['g_yaw']),
                        color=(0, 255, 0))

    new_img = draw_gaze(new_img, (img.shape[0] // 2 + 4, img.shape[1] // 2 + 4),
                        (entry['pred_pitch_cos'], entry['pred_yaw_weighted']),
                        color=(0, 0, 255))
    return new_img


def plot_df(df, rows=5, columns=10, img_size=2, idx_list=None, show_idx=True):
    """
    Given a dataframe containing session, person, frame, g_yaw, g_pitch, pred_yaw_weighted and pred_pitch_cos
    show the images with correct and predicted gazes.
    """
    if idx_list is None:
        idx_list = np.random.permutation(np.arange(len(df)))[:rows * columns]
        idx_list = [df.iloc[idx].name for idx in idx_list]
    else:
        rows = int(np.ceil(len(idx_list) / columns))
    _, ax = plt.subplots(figsize=(img_size * columns, rows * img_size), ncols=columns, nrows=rows)
    for i, idx in enumerate(idx_list):
        entry = df.loc[idx]
        img = draw_target_predicted_gaze(entry)
        if rows == 1:
            cur_ax = ax[i]
        else:
            cur_ax = ax[i // columns, i % columns]

        cur_ax.imshow(img)
        cur_ax.tick_params(left=False, right=False, top=False, bottom=False)
        cur_ax.axis('off')
        if show_idx:
            cur_ax.set_title(entry.name)
    plt.subplots_adjust(
        wspace=0.15,
        hspace=0.15,
    )
