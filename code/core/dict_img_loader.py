import numpy as np

from core.analysis_utils import get_frame, get_person, get_session
from core.metadata_parser import get_eye_bbox_dict
from data_loader import default_loader


class DictImgLoader:
    def __init__(self, data_dict, ignore_unavailable=False):
        """
        if ignore_unavailable is False, then the eye data must be present failing which it throws eror
        if ignore_unavailable is True, then if eye data is absent then it returns None in 'data' key

        """
        self._data_dict = data_dict
        self.ignore_unavailable = ignore_unavailable

    def get_val(self, fpath):
        session = get_session(fpath)
        person = get_person(fpath)
        frame = get_frame(fpath)
        if self.ignore_unavailable is False:
            return self._data_dict[session][person][frame]

        if session not in self._data_dict:
            return None
        if person not in self._data_dict[session]:
            return None
        if frame not in self._data_dict[session][person]:
            return None

        return self._data_dict[session][person][frame]

    def load(self, fpath):
        img = default_loader(fpath)
        return {'img': img, 'data': self.get_val(fpath)}


class DictEyeImgLoader(DictImgLoader):
    def __init__(self, mdata_path='/tmp2/ashesh/gaze360_data/metadata.mat', ignore_unavailable=False):
        super().__init__(get_eye_bbox_dict(mdata_path), ignore_unavailable=ignore_unavailable)

    def load(self, fpath):
        data_dict = super().load(fpath)
        data_dict['bbox'] = data_dict.pop('data')
        return data_dict


def scale_bbox(bbox: np.array, factor: float):
    """

    """
    x, y, w, h = bbox
    mid_x = x + w / 2
    mid_y = y + h / 2
    w_new = w * factor
    h_new = h * factor
    # NOTE: due to some incorrect labelling, this is needed to be done. instead of getting eye of the main person,
    # a side person's eye has been taken. this has resulted in x_new,y_new getting -ve
    x_new = max(0, mid_x - w_new / 2)
    y_new = max(0, mid_y - h_new / 2)
    return np.array([x_new, y_new, w_new, h_new], dtype=bbox.dtype)
