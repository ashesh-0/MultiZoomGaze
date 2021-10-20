import numpy as np

from core.dict_img_loader import scale_bbox


def test_scale_box():
    bbox = np.array([10.0, 25.0, 5, 5])
    factor = 2
    target_bbox = np.array([7.5, 22.5, 10, 10])
    scaled_bbox = scale_bbox(bbox, factor)
    assert max(np.abs(scaled_bbox - target_bbox)) < 1e-5

    bbox = np.array([0, 25.0, 5, 5])
    target_bbox = np.array([0, 22.5, 10, 10])
    scaled_bbox = scale_bbox(bbox, factor)
    assert max(np.abs(scaled_bbox - target_bbox)) < 1e-5
