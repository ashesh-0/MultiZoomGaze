import numpy as np
import torch
from mock import patch

from core.prediction_utils import get_prediction


def MockDataLoader(img_loader, *args, **kwargs):
    return img_loader


def get_mock_model_fn(*model_outputs):
    class MockModel:
        def __init__(self):
            self.eval_called_count = 0
            self._idx = 0

        def eval(self):
            self.eval_called_count += 1

        def __call__(self, img):
            if len(model_outputs) > 1:
                output = tuple([torch.Tensor(x[self._idx:self._idx + 1]) for x in model_outputs])
            else:
                output = torch.Tensor(model_outputs[0][self._idx:self._idx + 1])
            self._idx += 1
            return output

    model = MockModel()
    return model


@patch('core.prediction_utils.torch.utils.data.DataLoader', side_effect=MockDataLoader)
def test_get_prediction_multi_output(mock_dataloader):
    var = [
        [1.1, 1.1, 1.1],
        [2, 2, 2],
    ]
    desired_prediction = [
        [0, -0.5, -0.2],
        [-0.7, -0.9, 0],
    ]
    desired_target = torch.Tensor([
        [0.5, 0.5, -0.2],
        [-0.7, 0.9, 0],
    ])

    model = get_mock_model_fn(desired_prediction, var)
    img_loader = [(None, desired_target[0].reshape(1, -1)), (None, desired_target[1].reshape(1, -1))]
    prediction, target = get_prediction(model, img_loader, multi_output_model=True)
    assert np.max(np.abs(prediction - desired_prediction)) < 1e-6
    assert np.max(np.abs(target - desired_target.numpy())) < 1e-6


@patch('core.prediction_utils.torch.utils.data.DataLoader', side_effect=MockDataLoader)
def test_get_prediction_single_output(mock_dataloader):
    desired_prediction = [
        [0, -0.5, -0.2],
        [-0.7, -0.9, 0],
    ]
    desired_target = torch.Tensor([
        [0.5, 0.5, -0.2],
        [-0.7, 0.9, 0],
    ])

    model = get_mock_model_fn(desired_prediction)
    img_loader = [(None, desired_target[0].reshape(1, -1)), (None, desired_target[1].reshape(1, -1))]
    prediction, target = get_prediction(model, img_loader, multi_output_model=False)
    assert np.max(np.abs(prediction - desired_prediction)) < 1e-6
    assert np.max(np.abs(target - desired_target.numpy())) < 1e-6
