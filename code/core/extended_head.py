"""
This file allows one to get a zoomed out image of head. We use body's
"""
import os

import numpy as np
from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm

from core.analysis_utils import get_filename, get_frame, get_person, get_session
from core.metadata_parser import rescale_bbox
from core.pickle_utils import loadPickle, savePickle


class ExtendedHead:
    DICT_PKL = '/home/ashesh/data/extended_head_dicts_skip:{}.pkl'
    IMG_DIR = '/tmp2/ashesh/gaze360_extended_head/'

    def __init__(self,
                 img_fpath: str = None,
                 mdata_path: str = '/tmp2/ashesh/gaze360_data/metadata.mat',
                 skip_improper_imgs=True):
        """
        Args:
            img_fpath: Example:/home/ashesh/Documents/RA_code/Gaze360/code/validation.txt. If given, then it only
            retains images present in this file. This is used for doing train/val/test split.
            skip_improper_imgs: If True then all those images whose head version doesn't strictly lie within their body
            version, we skip them.  If False, then we trim the head bboxes of such images so that above constrained is
            made satisfied.
        """
        self._fpath = mdata_path
        self._mdata = loadmat(mdata_path)
        self._img_fpath = img_fpath
        self._skip_improper_imgs = skip_improper_imgs
        self._size_dict = None
        self._head_bbox_dict = None
        self._init_dicts()
        self._key_list = sorted([k for k in self._size_dict])
        if self._img_fpath is not None:
            self._filter_dicts()

    def _get(self, dictionary, session, person, frame):
        key = get_filename(session, person, frame, directory='', crop_type='head')
        return dictionary[key]

    def _add(self, dictionary, val, session, person, frame):
        key = get_filename(session, person, frame, directory='', crop_type='head')
        dictionary[key] = val

    def _filter_dicts(self):
        size_dict = {}
        head_bbox_dict = {}
        key_list = []
        N = len(self._key_list)
        skipped_N = 0
        with open(self._img_fpath, 'r') as f:
            for line in f:
                line = line[:-1]
                line = line.replace("\t", " ")
                line = line.replace("  ", " ")
                split_lines = line.split(" ")
                if (len(split_lines) > 3):
                    fpath = split_lines[0]
                    session = get_session(fpath)
                    person = get_person(fpath)
                    frame = get_frame(fpath)
                    key = get_filename(session, person, frame, directory='', crop_type='head')
                    if key in self._size_dict:
                        key_list.append(key)
                        size_dict[key] = self._size_dict[key]
                        head_bbox_dict[key] = self._head_bbox_dict[key]
                    else:
                        skipped_N += 1

        self._size_dict = size_dict
        self._head_bbox_dict = head_bbox_dict
        self._key_list = key_list
        new_N = len(self._key_list)
        print(f'[{self.__class__.__name__} Kept {self._img_fpath} data- {new_N} imgs, {round(new_N/N*100)}%'
              f' Skipped {skipped_N}')

    def _align_improper_imgs(self, head_bbox, body_bbox):
        # 13320 entries have head_bbox cropped left of body_bbox.
        left_head = head_bbox[:, 0] < body_bbox[:, 0]
        assert left_head.sum() == 13320
        head_bbox[left_head, 0] = body_bbox[left_head, 0]

        right_head = head_bbox[:, 2] > body_bbox[:, 2]
        assert right_head.sum() == 8657, f'{right_head.sum()}'
        head_bbox[right_head, 2] = body_bbox[right_head, 2]

        up_head = head_bbox[:, 1] < body_bbox[:, 1]
        assert up_head.sum() == 1
        head_bbox[up_head, 1] = body_bbox[up_head, 1]

    def _get_improper_filtr(self):
        """
        In some images, the width of the head crop is more than width of body crop. Such things lead to m > 1 which we
        don't want. So we are removing such images.
        """
        head_bbox = self._mdata['person_head_bbox']
        body_bbox = self._mdata['person_body_bbox']

        left_head = head_bbox[:, 0] < body_bbox[:, 0]
        right_head = head_bbox[:, 0] + head_bbox[:, 2] > body_bbox[:, 0] + body_bbox[:, 2]
        up_head = head_bbox[:, 1] < body_bbox[:, 1]

        improper_filter = np.logical_or(np.logical_or(left_head, right_head), up_head)
        # import pdb
        # pdb.set_trace()
        print(f'Improper filter: Left-{left_head.sum()} Right-{right_head.sum()} Up:{up_head.sum()}'
              f' All-{improper_filter.sum()}')
        return improper_filter

    def _init_dicts(self):
        if os.path.exists(self.__class__.DICT_PKL.format(self._skip_improper_imgs)):
            print('[ExtendedHead] Loading from pkl')
            data = loadPickle(self.__class__.DICT_PKL.format(self._skip_improper_imgs))
            self._size_dict = data['size_dict']
            self._head_bbox_dict = data['head_bbox_dict']
            return

        self._size_dict = {}
        self._head_bbox_dict = {}
        N = len(self._mdata['frame'][0])
        head_bbox = self._mdata['person_head_bbox']
        body_bbox = self._mdata['person_body_bbox']

        # len(set(np.where(improper == True)[0]).intersection(set(np.where(head_bbox[:, 3] == -1)[0])))
        # import pdb
        # pdb.set_trace()
        if self._skip_improper_imgs:
            improper = self._get_improper_filtr()
        else:
            self._align_improper_imgs(head_bbox, body_bbox)

        head_bbox = rescale_bbox(head_bbox, body_bbox)
        # import pdb
        # pdb.set_trace()
        for idx in tqdm(range(N)):
            if self._skip_improper_imgs and improper[idx]:
                continue
            session = self._mdata['recordings'][0, self._mdata['recording'][0, idx]][0]
            session = int(session.split('_')[1])
            person = self._mdata['person_identity'][0, idx]
            frame = self._mdata['frame'][0, idx]
            body_fpath = get_filename(session, person, frame, crop_type='body')
            self._add(self._size_dict, Image.open(body_fpath).size, session, person, frame)
            assert np.all(head_bbox[idx] >= 0), f'Issue at {idx}'
            self._add(self._head_bbox_dict, head_bbox[idx], session, person, frame)

        savePickle(
            self.__class__.DICT_PKL.format(self._skip_improper_imgs), {
                'size_dict': self._size_dict,
                'head_bbox_dict': self._head_bbox_dict
            })

    def _compute_xy_new(self, x, y, w, h, m, tx, ty):
        w_new = w * 1 / m
        h_new = h * 1 / m

        midx = x + w / 2
        midy = y + h / 2

        x_new = midx - w_new / 2
        y_new = midy - h_new / 2
        # TODO: check for +- with tx,ty
        x_new += tx
        y_new += ty
        return x_new, y_new, w_new, h_new

    def _get_ehead_bbox(self, session, person, frame, tx, ty, m):
        sizex, sizey = self._get(self._size_dict, session, person, frame)
        x, y, w, h = self._get(self._head_bbox_dict, session, person, frame)
        # tx, ty are with respect to head bbox. however, x,y,w,h is with respect to body. So here, we are converting
        # tx,ty to be with respect to body.
        tx = tx * w
        ty = ty * h

        midx = x + w / 2
        midy = y + h / 2

        x_new, y_new, w_new, h_new = self._compute_xy_new(x, y, w, h, m, tx, ty)
        # import pdb
        # pdb.set_trace()
        while x_new < 0:
            if tx < 0:
                tx = min(tx + 0.05 * w, 0)
                x_new, y_new, w_new, h_new = self._compute_xy_new(x, y, w, h, m, tx, ty)
            else:
                w_new = 2 * (midx + tx)
                m = w / w_new
                x_new, y_new, w_new, h_new = self._compute_xy_new(x, y, w, h, m, tx, ty)
                assert abs(x_new) < 1e-10, f'{x_new} is not 0'
                x_new = 0

        while y_new < 0:
            if ty < 0:
                ty = min(ty + 0.05 * h, 0)
                x_new, y_new, w_new, h_new = self._compute_xy_new(x, y, w, h, m, tx, ty)
            else:
                h_new = 2 * (midy + ty)
                m = h / h_new
                x_new, y_new, w_new, h_new = self._compute_xy_new(x, y, w, h, m, tx, ty)
                assert abs(y_new) <= 1e-10, f'{y_new} is not 0'
                y_new = 0

        while x_new + w_new > 1:
            if tx > 0:
                tx = max(0, tx - 0.05 * w)
                x_new, y_new, w_new, h_new = self._compute_xy_new(x, y, w, h, m, tx, ty)
            else:
                w_new = 2 * (1 - midx - tx)
                m = w / w_new
                x_new, y_new, w_new, h_new = self._compute_xy_new(x, y, w, h, m, tx, ty)
                assert abs(x_new + w_new - 1) < 1e-10, f'{x_new + w_new} is not 1'
                w_new = 1 - x_new

        while y_new + h_new > 1:
            if ty > 0:
                ty = max(0, ty - 0.05 * h)
                x_new, y_new, w_new, h_new = self._compute_xy_new(x, y, w, h, m, tx, ty)
            else:
                h_new = 2 * (1 - midy - ty)
                m = h / h_new
                x_new, y_new, w_new, h_new = self._compute_xy_new(x, y, w, h, m, tx, ty)
                assert abs(y_new + h_new - 1) < 1e-10
                h_new = 1 - y_new

        bbox = [int(x_new * sizex), int(y_new * sizey), int(w_new * sizex), int(h_new * sizey)]
        return {'bbox': bbox, 'tx': tx / w, 'ty': ty / h, 'm': m}

    def get(self, session, person, frame, tx, ty, m):
        # NOTE: tx=2 is needed in SpatialTransformer to totally displace the image. That has something to do with
        # affine_grid() using [-1,1] to create the grid. so it needs 2 to displace it. For emulating the same behaviour
        # here, we are dividing by 2 in both tx and ty
        bbox = self._get_ehead_bbox(session, person, frame, tx / 2, ty / 2, m)
        x, y, w, h = bbox['bbox']
        path = get_filename(session, person, frame, crop_type='head', directory=self.__class__.IMG_DIR)
        im = Image.open(path).convert('RGB')
        return {'img': im.crop((x, y, x + w, y + h)), 'tx': 2 * bbox['tx'], 'ty': 2 * bbox['ty'], 'm': bbox['m']}

    def get_from_index(self, index, tx, ty, m):
        fpath = self._key_list[index]
        session = get_session(fpath)
        person = get_person(fpath)
        frame = get_frame(fpath)

        return self.get(session, person, frame, tx, ty, m)

    def __len__(self):
        return len(self._key_list)


class ExtendedHeadLoader:
    def __init__(self, tx_max, ty_max, m_diff, img_fpath: str = None):
        assert tx_max >= 0
        assert ty_max >= 0

        self._tx = tx_max
        self._ty = ty_max
        self._m = m_diff
        self.loader = ExtendedHead(img_fpath=img_fpath)

    def get(self, fpath):
        session = get_session(fpath)
        person = get_person(fpath)
        frame = get_frame(fpath)
        m = 1 + self._m * np.random.rand()
        tx = 2 * self._tx * np.random.rand() - self._tx
        ty = 2 * self._ty * np.random.rand() - self._ty
        return self.loader.get(session, person, frame, tx, ty, m)

    def __getitem__(self, idx):
        m = 1 + self._m * np.random.rand()
        tx = 2 * self._tx * np.random.rand() - self._tx
        ty = 2 * self._ty * np.random.rand() - self._ty
        return self.loader.get_from_index(idx, tx, ty, m)

    def __call__(self, fpath):
        return self.get(fpath)['img']

    def __len__(self):
        return len(self.loader)
