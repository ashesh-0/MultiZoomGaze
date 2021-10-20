
import numpy as np
import pandas as pd
import torch

from core.gaze_utils import average_angle, compute_angular_error_arr
from core.prediction_utils import get_df_from_predictions, get_prediction_with_uncertainty
from sinecosine_model.train_utils import get_cosinebased_yaw_pitch, get_sinebased_yaw_pitch


def compute_angular_error_sine_and_cosine_arr(input_: np.ndarray, target_: np.ndarray) -> np.ndarray:
    input_ = torch.Tensor(input_)
    target_ = torch.Tensor(target_)
    yaw_pitch_cosine = get_cosinebased_yaw_pitch(input_)
    yaw_pitch_sine = get_sinebased_yaw_pitch(input_)

    pred_yaw = average_angle(yaw_pitch_cosine[:, 0], yaw_pitch_sine[:, 0]).view(-1, 1)
    pred_pitch = average_angle(yaw_pitch_cosine[:, 1], yaw_pitch_sine[:, 1]).view(-1, 1)
    pred = torch.cat([pred_yaw, pred_pitch], dim=1)

    target = get_cosinebased_yaw_pitch(target_)
    return compute_angular_error_arr(np.array(pred), np.array(target))


def get_df_from_dataloader(model,
                           img_loader,
                           multi_output_model: bool = True,
                           target_columns=['sin(Yaw)', 'cos(Yaw)', 'sin(Pitch)'],
                           compute_angular_error_fn=compute_angular_error_sine_and_cosine_arr,
                           lstm_model: bool = False,
                           num_workers: int = 5) -> pd.DataFrame:
    """
    Args:
        multi_output_model: if true then we take the first prediction. If false then whole prediction is taken.
            Generally, model predict gaze along with uncertainity. In that case, we want to just get gaze.
            However, for boolean classifier, there is just single output.

        lstm_model: If True, then it means that img_loader contains a sequence of filepaths for each entry. If False,
            then it contains just one filepath of the image for each entry.

    """
    assert multi_output_model is True
    prediction, actual, uncertainty = get_prediction_with_uncertainty(
        model, img_loader, multi_output_model=multi_output_model, num_workers=num_workers)

    df = get_df_from_predictions(
        prediction,
        actual,
        img_loader,
        target_columns=target_columns,
        compute_angular_error_fn=compute_angular_error_fn,
        lstm_model=lstm_model)
    df['uncertainty'] = uncertainty
    return df
