import math
import os
import pickle
from typing import List

import pandas as pd

import matplotlib.pyplot as plt


def get_session(filename):
    """
    Example of filename: '/tmp2/ashesh/gaze360_data/imgs/rec_022/head/000000/000131.jpg'
    """

    tokens = filename.split('/')
    return int(tokens[-4].split('_')[1])


def get_person(filename):
    """
    Example of filename: '/tmp2/ashesh/gaze360_data/imgs/rec_022/head/000000/000131.jpg'
    """

    tokens = filename.split('/')
    return int(tokens[-2])


def get_frame(filename):
    """
    Example of filename: '/tmp2/ashesh/gaze360_data/imgs/rec_022/head/000000/000131.jpg'
    """

    tokens = filename.split('/')
    return int(tokens[-1].split('.')[0])


def print_gaze360_metrics(df, target_col='angular_err_avg'):
    """
    It prints the metrics shown in Gaze360 paper.
    """
    assert 'g_yaw' in df.columns
    assert target_col in df.columns
    assert 'g_z' in df.columns

    front180_df = df[df['g_z'] <= 0]
    back_df = df[df['g_z'] > 0]

    frontfacing_df = df[df.g_yaw.abs() * 180 / math.pi <= 20]
    error_uncertainty_correlation = df[['uncertainty', target_col]].corr().loc['uncertainty', target_col]
    error_uncertainty_rank_correlation = df[['uncertainty', target_col]].corr(method='spearman').loc['uncertainty',
                                                                                                     target_col]

    def mean_std_str(df: pd.Series):
        return f'{round(df.mean(),2)} +- {round(df.std(),2)}'

    data = {
        'overall': mean_std_str(df[target_col]),
        'front90': mean_std_str(front180_df[target_col]),
        'front20': mean_std_str(frontfacing_df[target_col]),
        'back': mean_std_str(back_df[target_col])
    }
    print('')
    print('[Error Column]', target_col)
    print('[Overall]', data['overall'])
    print('[Front (+-90 Yaw)]', data['front90'])
    print('[FrontFacing (+-20 Yaw)]', data['front20'])
    print('[Back]', data['back'])
    print('[Uncert. Corr]', round(error_uncertainty_correlation, 2))
    print('[Uncert. Corr Rank]', round(error_uncertainty_rank_correlation, 2))
    return data


def get_filename(session: int,
                 person: int,
                 frame: int,
                 crop_type: str = 'body',
                 directory='/tmp2/ashesh/gaze360_data/imgs'):
    """
    For session=22,person=0 and frame = 131,
    example of filename: '/tmp2/ashesh/gaze360_data/imgs/rec_022/head/000000/000131.jpg'
    """
    assert crop_type in ['head', 'body']
    return os.path.join(directory, f"rec_{int(session):03}/{crop_type}/{int(person):06}/{int(frame):06}.jpg")


def angular_analysis(df, target_columns, pitch_buckets=10, yaw_buckets=10):
    assert 'g_yaw' in df.columns
    assert 'g_pitch' in df.columns
    # assert 'angular_err' in df.columns
    df['yaw_bucket'] = pd.qcut(df['g_yaw'], yaw_buckets).apply(lambda x: round(x.mid, 2)).astype(float)
    df['pitch_bucket'] = pd.qcut(df['g_pitch'], yaw_buckets).apply(lambda x: round(x.mid, 2)).astype(float)

    return {
        'yaw': df.groupby('yaw_bucket')[target_columns].mean(),
        'pitch': df.groupby('pitch_bucket')[target_columns].mean()
    }


def plot_angular_analysis_dicts(data_dict, figsize=(15, 5)):

    _, ax = plt.subplots(figsize=figsize, ncols=2)
    for key in sorted(data_dict.keys()):
        data_dict[key]['pitch'].to_frame(key).plot(marker='o', ax=ax[0])
        data_dict[key]['yaw'].to_frame(key).plot(marker='o', ax=ax[1])

    ax[0].set_title('Variation of angular error with pitch')
    ax[1].set_title('Variation of angular error with yaw')
    ax[0].set_ylabel('Angular error')

    ax[0].set_xlabel('Pitch (in radian)')
    ax[1].set_xlabel('Yaw (in radian)')

    return data_dict


def plot_angular_analysis_paths(pkl_paths: List[str], figsize=(15, 5)):
    data_dict = {}
    for pkl_path in pkl_paths:
        with open(pkl_path, 'rb') as f:
            data_dict[pkl_path] = pickle.load(f)

    return plot_angular_analysis_dicts(data_dict, figsize=figsize)


def count_parametersM(model):
    c = sum(p.numel() for p in model.parameters() if p.requires_grad)
    c = c / 10**6
    return round(c, 1)
