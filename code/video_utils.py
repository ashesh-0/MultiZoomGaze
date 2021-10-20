import os

import cv2

from core.analysis_utils import get_filename, get_frame


def add_frame(img, fpath):
    font = cv2.FONT_HERSHEY_DUPLEX
    fontScale = 0.5
    fontColor = (0, 0, 255)
    lineType = 1
    cv2.putText(img, f'F:{round(get_frame(fpath))}', (img.shape[0] // 10, int(img.shape[1] / 1.1 - 10)), font,
                fontScale, fontColor, lineType)


def add_error(img, error):
    font = cv2.FONT_HERSHEY_DUPLEX
    fontScale = 0.5
    fontColor = (255, 0, 0)
    lineType = 1
    cv2.putText(img, f'Error:{round(error)}', (img.shape[0] // 10, img.shape[1] // 8), font, fontScale, fontColor,
                lineType)


def create_img_array(fpaths, errors=None):
    assert errors is None or len(fpaths) == len(errors)
    img_array = []
    size = None
    for i, fpath in enumerate(fpaths):
        img = cv2.imread(fpath)
        height, width, layers = img.shape
        if size is None:
            size = (width, height)
            print('Size:', size)
        else:
            img = cv2.resize(img, size)

        if errors is not None:
            add_error(img, errors[i])
        add_frame(img, fpath)
        img_array.append(img)
    return img_array


def create_movie_from_array(img_array, movie_fname):
    size = img_array[0].shape[:2][::-1]
    # import pdb
    # pdb.set_trace()
    out = cv2.VideoWriter(movie_fname, cv2.VideoWriter_fourcc(*'mp4v'), 2, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def create_movie(fpaths, errors, movie_fname):
    img_array = create_img_array(fpaths, errors)
    create_movie_from_array(img_array, movie_fname)


def create_movie_from_df(df,
                         session,
                         person,
                         start_frame,
                         end_frame,
                         error_col,
                         folder='videos/',
                         crop_type='head',
                         fileprefix=''):

    if start_frame is None or end_frame is None:
        if start_frame is not None:
            frame_filter = df.frame >= start_frame
        elif end_frame is not None:
            frame_filter = df.frame <= end_frame
        else:
            frame_filter = True
    else:
        f_list = list(range(start_frame, end_frame + 1))
        frame_filter = df.frame.isin(f_list)

    df = df[(df.session == session) & (df.person == person) & frame_filter]
    df = df.sort_values('frame')

    if len(fileprefix) > 0:
        movie_fname = os.path.join(folder, f'{fileprefix}_{session}_{person}_{start_frame}-{end_frame}.mp4')
    else:
        movie_fname = os.path.join(folder, f'{session}_{person}_{start_frame}-{end_frame}.mp4')

    fpaths = []
    errors = []
    for _, row in df.iterrows():
        fpath = get_filename(session, person, row['frame'], crop_type=crop_type)
        fpaths.append(fpath)
        errors.append(row[error_col])

    create_movie(fpaths, errors, movie_fname)
    return movie_fname
