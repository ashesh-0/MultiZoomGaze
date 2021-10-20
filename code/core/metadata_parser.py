from typing import Dict

import numpy as np
from scipy.io import loadmat


def rescale_eye(eye: np.array, head_bbox: np.array) -> np.array:
    """
    In metadata.mat, the person_eye_left_box or person_eye_left_box contains eye x,y,w,h data for both eyes.
    These coordinates are in normalized form with respect to original full scale image.
    This function converts them to normalized form with respect to head crop as defined by head_bbox. head_bbox is also
    in normalized coordinates with respect to original full scale image.

    """
    return rescale_bbox(eye, head_bbox)


def rescale_bbox(bbox, enclosing_bbox):
    invalid_entries = np.all(bbox.astype(int) == -1, axis=1)
    bbox = [
        (bbox[:, 0] - enclosing_bbox[:, 0]) * (1 / enclosing_bbox[:, 2]),
        (bbox[:, 1] - enclosing_bbox[:, 1]) * (1 / enclosing_bbox[:, 3]),
        bbox[:, 2] / enclosing_bbox[:, 2],
        bbox[:, 3] / enclosing_bbox[:, 3],
    ]
    bbox = np.vstack(bbox).T
    bbox[invalid_entries, :] = -1
    print('Extra invalid entries', np.any(bbox[~invalid_entries, :] < 0, axis=1).sum())
    bbox[np.any(bbox < 0, axis=1), :] = -1
    return bbox


def get_eye_bbox_dict(
        mdata_path: str = '/tmp2/ashesh/gaze360_data/metadata.mat') -> Dict[int, Dict[int, Dict[int, dict]]]:
    """
    Returns a nested dict of all integer keys session=> person => frame
    To get left eye bounding box, one needs to do :
        session = 0
        person = 12
        frame = 10
        data_dict[session][person][frame]['left']
    """
    mdata = loadmat(mdata_path)
    leye = mdata['person_eye_left_bbox']
    reye = mdata['person_eye_right_bbox']

    head_bbox = mdata['person_head_bbox']
    leye = rescale_eye(leye, head_bbox)
    reye = rescale_eye(reye, head_bbox)

    sessions = mdata['recording']
    persons = mdata['person_identity']
    frames = mdata['frame']
    index_data = np.concatenate([sessions, persons, frames], axis=0).T
    bbox_data = np.concatenate([leye, reye], axis=1)

    data_dict = {}
    session_list = np.unique(mdata['recording'][0])
    for session in session_list:
        data_dict[session] = {}
        s_f = index_data[:, 0] == session
        s_index_data = index_data[s_f]
        s_bbox_data = bbox_data[s_f]
        persons = np.unique(s_index_data[:, 1])
        for person in persons:
            data_dict[session][person] = {}
            p_f = s_index_data[:, 1] == person
            p_index_data = s_index_data[p_f]
            p_bbox_data = s_bbox_data[p_f]
            for f_idx_data, f_bbox_data in zip(p_index_data, p_bbox_data):
                data_dict[session][person][f_idx_data[2]] = {'left': f_bbox_data[:4], 'right': f_bbox_data[4:]}
    return data_dict
