from typing import Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from core.analysis_utils import get_frame, get_person, get_session
from core.gaze_utils import compute_angular_error_arr


def get_prediction_with_uncertainty(model,
                                    img_loader,
                                    num_workers: int = 2,
                                    batch_size: int = 64) -> Tuple[np.array, np.array]:
    test_loader = torch.utils.data.DataLoader(img_loader,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              pin_memory=False)

    actual = []
    prediction = []
    uncertainty = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # import pdb
            # pdb.set_trace()
            gaze = batch[-1]
            inp = tuple(batch[:-1])
            pred = model(*inp)
            pred_gaze, pred_uncert = pred
            prediction.append(pred_gaze.cpu().numpy())
            actual.append(gaze.cpu().numpy())
            uncertainty.append(pred_uncert.cpu().numpy())

    prediction = np.concatenate(prediction, axis=0)
    actual = np.concatenate(actual, axis=0)
    uncertainty = np.concatenate(uncertainty, axis=0)
    return (prediction, actual, uncertainty)


def get_prediction(model,
                   img_loader,
                   multi_output_model: bool = True,
                   num_workers: int = 2,
                   batch_size: int = 64) -> Tuple[np.array, np.array]:
    """
    Args:
        multi_output_model: if true then we take the first prediction. If false then whole prediction is taken.
    Generally, model predict gaze along with uncertainity. In that case, we want to just get gaze.
    However, for boolean classifier, there is just single output.
    """
    test_loader = torch.utils.data.DataLoader(img_loader,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              pin_memory=False)

    actual = []
    prediction = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            gaze = batch[-1]
            inp = tuple(batch[:-1])
            pred = model(*inp)
            if multi_output_model:
                pred = pred[0]
            prediction.append(pred.cpu().numpy())
            actual.append(gaze.cpu().numpy())

    prediction = np.concatenate(prediction, axis=0)
    actual = np.concatenate(actual, axis=0)
    return (prediction, actual)


def get_df_from_predictions(prediction: np.array,
                            actual: np.array,
                            img_loader,
                            target_columns=['yaw', 'pitch'],
                            compute_angular_error_fn=compute_angular_error_arr,
                            lstm_model: bool = False) -> pd.DataFrame:
    """
    For gaze predictions
    Args:
        lstm_model: If True, then it means that img_loader contains a sequence of filepaths for each entry. If False,
            then it contains just one filepath of the image for each entry.
    """
    fpath_gaze_tuples = img_loader.imgs
    mdata = []
    for i, (fpath, gaze) in enumerate(fpath_gaze_tuples):
        if lstm_model is True:
            assert isinstance(fpath, list)
            fpath = fpath[len(fpath) // 2]

        a1 = [get_session(fpath), get_person(fpath), get_frame(fpath)]

        mdata.append(a1 + actual[i].tolist() + prediction[i].tolist() + gaze.tolist())

    pred_cols = [f'pred_{c}' for c in target_columns]
    act_cols = [f'g_{c}' for c in target_columns]
    mdata_df = pd.DataFrame(mdata,
                            columns=['session', 'person', 'frame'] + act_cols + pred_cols + ['g_x', 'g_y', 'g_z'])

    mdata_df['angular_err'] = compute_angular_error_fn(mdata_df[pred_cols].values, mdata_df[act_cols].values)

    mdata_df.index.name = 'idx'
    mdata_df.reset_index(inplace=True)
    return mdata_df


def get_df_from_dataloader(model,
                           img_loader,
                           multi_output_model: bool = True,
                           lstm_model: bool = False,
                           target_columns=['yaw', 'pitch'],
                           compute_angular_error_fn=compute_angular_error_arr,
                           num_workers: int = 5) -> pd.DataFrame:
    """
    model should predict gaze.
    Args:
        multi_output_model: if true then we take the first prediction. If false then whole prediction is taken.
            Generally, model predict gaze along with uncertainity. In that case, we want to just get gaze.
            However, for boolean classifier, there is just single output.

        lstm_model: If True, then it means that img_loader contains a sequence of filepaths for each entry. If False,
            then it contains just one filepath of the image for each entry.

    """
    prediction, actual = get_prediction(model,
                                        img_loader,
                                        multi_output_model=multi_output_model,
                                        num_workers=num_workers)
    return get_df_from_predictions(
        prediction,
        actual,
        img_loader,
        lstm_model=lstm_model,
        target_columns=target_columns,
        compute_angular_error_fn=compute_angular_error_fn,
    )
