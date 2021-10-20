import numpy as np

from PIL import Image

STREET_IMG = Image.open("/home/ashesh/code/MultiZoomGaze360/dataset/WildImgCrops/pexels-lukas-kloeppel-2416653.jpg")
STREET_IMG_BBOX_FPATH = '/home/ashesh/code/MultiZoomGaze360/dataset/WildImgCrops/bounding_boxes.pkl'


def get_bbox(idx, bbox_data):
    r = bbox_data['bbox'][idx]
    r = bbox_data['factor'] * np.array(r)
    print(r)

    imCrop = np.array(STREET_IMG)[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    return imCrop
