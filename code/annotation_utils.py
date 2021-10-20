"""
For doing annotation, we want to get several properties like session, person, angular error etc with the image path.
In this file, given a prediction text file and metadata df, it returns a dataframe with each row belonging to one image
"""
import math

import numpy as np
import pandas as pd
from scipy.io import loadmat

from core.analysis_utils import get_frame, get_person, get_session
from core.gaze_utils import compute_angular_error_arr


def get_mdata(metadata_fpath: str) -> pd.DataFrame:
    """
    Takes in the filepath provided with gaze360 dataset: metadata.mat and returns dataframe.
    """
    mdata = loadmat(metadata_fpath)
    mdata_df = pd.DataFrame(mdata['recording'].T, columns=['session'])
    mdata_df['person'] = mdata['person_identity'][0].astype(np.int32)
    mdata_df['frame'] = mdata['frame'][0].astype(np.int32)
    mdata_df['eye_target_dist'] = np.linalg.norm(
        mdata['target_pos3d'] - mdata['person_eyes3d'], axis=1).astype(np.float32)
    mdata_df['eye_camera_dist'] = np.linalg.norm(mdata['person_eyes3d'], axis=1).astype(np.float32)
    mdata_df['camera_idx'] = mdata['person_cam'][0]
    return mdata_df


def get_df(path: str, metadata_fpath: str = '/tmp2/ashesh/gaze360_data/metadata.mat') -> pd.DataFrame:
    """
    Args:
        path: it is a text file where predictions have been saved. Generally there is separate file for train,
            validation and test.
        metadata_fpath: .mat file provided by the Gaze360 dataset.
    Returns:
        returned dataframe has same size as number of entries in `path`.
    """
    mdata_df = get_mdata(metadata_fpath=metadata_fpath)
    df = pd.read_csv(path)
    df['frame'] = df['file'].apply(get_frame).astype(np.int32)
    df['session'] = df['file'].apply(get_session).astype(np.int32)
    df['person'] = df['file'].apply(get_person).astype(np.int32)

    df['angular_err'] = compute_angular_error_arr(df[['pred_yaw', 'pred_pitch']].values, df[['g_yaw',
                                                                                             'g_pitch']].values)

    df['g_z_bin'] = pd.cut(df['g_z'], 10).apply(lambda x: x.mid)
    df['g_z_bin'] = df['g_z_bin'].astype(float)

    df['g_yaw_bin'] = pd.cut(df['g_yaw'], 10).apply(lambda x: x.mid)
    df['g_yaw_bin'] = df['g_yaw_bin'].astype(float)

    df['g_pitch_bin'] = pd.cut(df['g_pitch'], 10).apply(lambda x: x.mid)
    df['g_pitch_bin'] = df['g_pitch_bin'].astype(float)

    df['g_ec_angle'] = (-1 * df['g_z']).apply(math.acos)

    df_with_dist = pd.merge(df, mdata_df, how='left', on=['session', 'person', 'frame'])
    df_with_dist['ec_dist_bin'] = pd.cut(df_with_dist['eye_camera_dist'], 20).apply(lambda x: x.mid)
    df_with_dist['ec_dist_bin'] = df_with_dist['ec_dist_bin'].astype(float)

    return df_with_dist
