from typing import List

import numpy as np
import pandas as pd


def _rolling_mean(a: np.ndarray, n: int):
    n = min(a.shape[0], n)
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    ret[n - 1:] = ret[n - 1:] / n

    if len(ret.shape) == 1:
        ret = ret.reshape(-1, 1)

    ret[:n - 1] = ret[:n - 1] / np.arange(1, n).reshape(n - 1, 1)
    return ret


def rolling_mean(df: pd.DataFrame, group_cols: List[str], target_cols: List[str], period: int) -> pd.DataFrame:
    orig_index = df.index.copy()
    df = df.sort_values(group_cols)
    end_idx = df.groupby(group_cols).size().cumsum().values

    data = df[target_cols].values
    output = np.zeros((df.shape[0], len(target_cols)))
    s = 0
    for e in end_idx:
        output[s:e, :] = _rolling_mean(data[s:e, :], period)
        s = e

    output_df = pd.DataFrame(output, index=df.index, columns=[f'{c}_smooth' for c in target_cols])
    return output_df.loc[orig_index]


def get_missing_frames_df(data_df: pd.DataFrame, missing_threshold: int = 1) -> pd.DataFrame:
    req_cols = ['session', 'person', 'frame']
    assert set(data_df.columns).intersection(set(req_cols)) == set(req_cols)

    N = data_df.shape[0]
    index = data_df.index.copy()
    data_df = data_df.sort_values(['session', 'person', 'frame'])
    skips = np.zeros((N, 1), dtype=bool)

    end_idx = data_df.groupby(['session', 'person']).size().cumsum().values
    frame = data_df['frame'].values
    s = 0
    for e in end_idx:
        sp_frame = frame[s:e]
        diff_frame = sp_frame[1:] - sp_frame[:-1]
        assert not np.any(diff_frame <= 0)
        skips[s + 1:e, 0] = diff_frame >= 1 + missing_threshold
        s = e

    print(f'{round(100*skips.sum()/N,2)}% entries have missing frames')
    return pd.DataFrame(skips, columns=['missing_frame'], index=data_df.index).loc[index]
