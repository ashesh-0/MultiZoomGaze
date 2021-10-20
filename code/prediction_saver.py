
"""
Functions in this file is used to predict and then save to the file.
Format of the file is identical to train.txt,test.txt and validate.txt
"""
from typing import List

import torch
from tqdm import tqdm


def append_to_file(filehandle, imagefilenames: List[str], xyz_gaze: List[List[float]], target: List[List[float]],
                   prediction: List[List[float]]) -> None:
    """
    This function appends prediction data in a comma separated format to an opened file.
    Args:
        filehandle: filehandle where to append the data.
        imagefilenames: List of full path of image files.
        xyz_gaze: List of gaze in XYZ format for each of the image.
        target: List of gaze in yaw pitch format for each image.
        prediction: List of predictions in yaw pitch format for each image.

    """

    def arr_to_str(arr):
        return ','.join([str(x) for x in arr])

    output_str = ''
    for fn, gz, tr, pr in zip(imagefilenames, xyz_gaze, target, prediction):
        output_str += f'{fn},{arr_to_str(gz)},{arr_to_str(tr)},{arr_to_str(pr)}\n'
    filehandle.write(output_str)


def save_predictions(fpath: str, img_loader: torch.utils.data.Dataset, model, batch_size: int, workers: int = 2):
    """
    This functions creates a file at path fpath and saves in it the image paths, actual gaze in xyz and yaw pitch format
    and predictions made by the model.
    Args:
        fpath: predictions will be saved to this file. older data will be overwritten.
        img_loader: image loader which returns the images along with target variable.
        model: pytorch model with pre-loaded learnt weights.
    """
    model.eval()
    data_loader = torch.utils.data.DataLoader(
        img_loader, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=False)

    with open(fpath, 'w') as filehandle:
        filehandle.write('file,g_x,g_y,g_z,g_yaw,g_pitch,pred_yaw,pred_pitch\n')

        for i, (source_frame, target) in enumerate(tqdm(data_loader)):

            source_frame = source_frame.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            source_frame_var = torch.autograd.Variable(source_frame)
            with torch.no_grad():
                # compute output
                prediction, _ = model(source_frame_var)
                prediction = [list(x) for x in prediction.cpu().data.numpy()]
                s_idx = i * batch_size
                e_idx = min(len(img_loader.imgs), (i + 1) * batch_size)

                # each has 7 images, pick the center (4th) image.
                fnames = [img_loader.imgs[idx][0][3] for idx in range(s_idx, e_idx)]
                # xyz gaze.
                raw_gazes = [img_loader.imgs[idx][1] for idx in range(s_idx, e_idx)]
                targets = [list(x) for x in target.cpu().data.numpy()]

                append_to_file(filehandle, fnames, raw_gazes, targets, prediction)
