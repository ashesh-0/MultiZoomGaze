"""
Data sampler where the criteria is the tilt angle of eyebbox and area of eye bbox.
"""
import math
import os
import pickle
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from core.eye_crop_transformation import FixedSizeCropTransformation
from core.paired_datasampler import DataSource, Levels, OneLevel


class DatasetEyeIndexSampler:
    def __init__(
        self,
        dataset,
        tilt_levels=None,
        len_levels=None,
        ignore_nbr_cnt=0,
        sample_nbr_cnt=1,
        magnify_eye_region_factor=1.5,
        use_same_bucket=True,
        data_source=DataSource.XGaze,
    ):
        """

        We have dividied the yaw and pitch angles into discrete bins of sizes yaw_bin_size and pitch_bin_size resp.
        Given a point in a bin B,
            a. If use_same_bucket is True then we sample another point from same bin.
            b. If use_same_bucket is False, then we sample another point from a neighbouring bin.
                * If We ignore ignore_nbr_cnt number of bins on either side of the original bin.
                * We sample a bin randomly from next sample_nbr_cnt bins on either side of the original bin B.
        Args:
            ignore_nbr_cnt: Ignore this many number of buckets in the neightbourhood on both sides.
            sample_nbr_cnt: After ignore region, sample from these many regions.
            use_same_bucket: Whether to ignore same bucket or not.
        """
        self._dataset = dataset
        self._tilt = tilt_levels
        self._len = len_levels
        if self._tilt is None:
            # quantile levels on validation set of Gaze360.
            # tilt_levels = [-90, -29.8, -6.4, -4.3, -2.8, -1.7, -0.6, 0.5, 1.7, 3.2, 5.6, 90]
            tilt_levels = [
                -90, -29.8, -8.6, -6.4, -5.3, -4.3, -3.4, -2.8, -2.3, -1.7, -1.2, -0.6, 0., 0.5, 1.1, 1.7, 2.4, 3.2,
                4.1, 5.6, 7.9, 90
            ]
            self._tilt = Levels(-1, 90, -90, levels=tilt_levels)
        if self._len is None:
            # quantile levels on validation set of Gaze360.
            len_levels_dict = {
                1.0: [0, 10.5, 16., 17.5, 19., 20.5, 23.5, 26.5, 33.5, 41., 52.5, 224],
                1.5: [0, 13., 20., 22., 24., 25.5, 29.5, 33.5, 42., 52., 66., 224],
                2.0: [0, 16., 25., 27., 28.5, 30.5, 35.5, 40., 51., 62.5, 79.5, 224],
                3.0: [0, 22., 33., 36.5, 39., 41.5, 48., 53.5, 68., 84.5, 106.5, 224],
            }
            self._len = Levels(-1, 224, 0, levels=len_levels_dict[magnify_eye_region_factor])

        self.leveldata = {}
        self.idxdata = []
        self._use_same_bucket = use_same_bucket
        self._ign_cnt = ignore_nbr_cnt
        self._sample_nbr_cnt = sample_nbr_cnt
        self._data_source = data_source
        self.eye_region = FixedSizeCropTransformation(magnify_eye_region_factor=magnify_eye_region_factor)
        print(f'[{self.__class__.__name__}] Fac:{magnify_eye_region_factor}')

    def pickle_fpath(self):
        src = 'XG' if self._data_source == DataSource.XGaze else 'Gz360'
        path = (f'EyeDISampler_{src}_{len(self._dataset)}_T:{len(self._tilt)}_L:{len(self._len)}_'
                f'MF:{self.eye_region._magnify_eye_region_factor}.pkl')
        return path

    def load(self):
        print(f'[{self.__class__.__name__}] Loading from {self.pickle_fpath()}')
        with open(self.pickle_fpath(), 'rb') as f:
            data = pickle.load(f)
            self.leveldata = data['leveldata']
            self.idxdata = data['idxdata']

    def save(self):
        print(f'[{self.__class__.__name__}] Saving to {self.pickle_fpath()}')
        with open(self.pickle_fpath(), 'wb') as f:
            pickle.dump({'leveldata': self.leveldata, 'idxdata': self.idxdata}, f)

    def get_data_dict(self, index):
        path_source, gaze = self._dataset.imgs[index]
        dic = self._dataset.loader(path_source)
        dic['bbox'] = deepcopy(dic['bbox'])
        return dic

    def get_tilt(self, leye_b, reye_b):
        if any(leye_b < 0) or any(reye_b < 0):
            return None

        assert -1 not in leye_b
        assert -1 not in reye_b
        diff = leye_b[:2] - reye_b[:2]
        if abs(diff[0]) < 1e-6:
            diff[0] = 1e-6
        return np.arctan(diff[1] / diff[0])

    def get_length(self, data_dict):
        x_min, x_max, y_min, y_max = self.eye_region.get_combined_eyes_pixel_coordinates(data_dict)
        eyes_len = ((x_max - x_min) + (y_max - y_min)) / 2
        return eyes_len

    def create(self):
        if os.path.exists(self.pickle_fpath()):
            print(f'[{self.__class__.__name__}] Loading from {self.pickle_fpath()}')
            self.load()
            return
        for i in tqdm(range(len(self._dataset))):
            data_dict = self.get_data_dict(i)
            # (x_min, x_max, y_min, y_max)
            leye_b, reye_b = self.eye_region.get_pixel_coordinates(data_dict, np.array(data_dict['img']).shape)
            tilt_i = self.get_tilt(leye_b, reye_b)

            if tilt_i is None:
                self.idxdata.append((None, None))
                continue
            len_i = self.get_length(data_dict)
            y_idx = self._tilt.get_idx(tilt_i)
            p_idx = self._len.get_idx(len_i)

            self.idxdata.append((y_idx, p_idx))

            if y_idx not in self.leveldata:
                self.leveldata[y_idx] = {}
            if p_idx not in self.leveldata[y_idx]:
                y_range = self._tilt[y_idx]
                p_range = self._len[p_idx]
                self.leveldata[y_idx][p_idx] = OneLevel(y_range, p_range)

            self.leveldata[y_idx][p_idx].add(i)
        self.save()

    def nonempty_tilt_idx(self, idx):
        if idx not in self.leveldata or not self.leveldata[idx]:
            return False
        return True

    def nonempty_len_idx(self, idx, yaw_idx):
        return idx in self.leveldata[yaw_idx]

    def next_tilt_idx(self, idx):
        N = len(self._yaw)
        for next_idx in range(idx + 1, N):
            if self.nonempty_tilt_idx(next_idx):
                return next_idx
        return None

    def previous_tilt_idx(self, idx):
        for prev_idx in range(idx - 1, -1, -1):
            if self.nonempty_tilt_idx(prev_idx):
                return prev_idx
        return None

    def next_len_idx(self, idx, yaw_idx):
        N = len(self._pitch)
        for next_idx in range(idx + 1, N):
            if self.nonempty_len_idx(next_idx, yaw_idx):
                return next_idx

        return None

    def previous_len_idx(self, idx, yaw_idx):
        for prev_idx in range(idx - 1, -1, -1):
            if self.nonempty_len_idx(prev_idx, yaw_idx):
                return prev_idx
        return None

    def neighbour_tilt_idx(self, idx, direction=0, ignore_count=None, neighbour_count=None):
        assert self._use_same_bucket is False
        if ignore_count is None:
            ignore_count = self._ign_cnt
            assert neighbour_count is None
            neighbour_count = self._sample_nbr_cnt

        diff_start = 1 + ignore_count
        diff_end = diff_start + neighbour_count
        idx_list = []
        if direction >= 0:
            n_idx = self.next_tilt_idx(idx + diff_start - 1)
            while n_idx is not None and n_idx < idx + diff_end:
                idx_list.append(n_idx)
                n_idx = self.next_tilt_idx(n_idx)

        if direction <= 0:
            p_idx = self.previous_tilt_idx(idx - diff_start + 1)
            while p_idx is not None and p_idx > idx - diff_end:
                idx_list.append(p_idx)
                p_idx = self.previous_tilt_idx(p_idx)

        return np.random.choice(idx_list) if len(idx_list) > 0 else None

    def neighbour_len_idx(self, idx, yaw_idx, direction=0, ignore_count=None, neighbour_count=None):
        assert self._use_same_bucket is False
        if ignore_count is None:
            ignore_count = self._ign_cnt
            assert neighbour_count is None
            neighbour_count = self._sample_nbr_cnt

        diff_start = 1 + ignore_count
        diff_end = diff_start + neighbour_count
        idx_list = []
        if direction >= 0:
            n_idx = self.next_len_idx(idx + diff_start - 1, yaw_idx)
            while n_idx is not None and n_idx < idx + diff_end:
                idx_list.append(n_idx)
                n_idx = self.next_len_idx(n_idx, yaw_idx)

        if direction <= 0:
            p_idx = self.previous_len_idx(idx - diff_start + 1, yaw_idx)
            while p_idx is not None and p_idx > idx - diff_end:
                idx_list.append(p_idx)
                p_idx = self.previous_len_idx(p_idx, yaw_idx)

        return np.random.choice(idx_list) if len(idx_list) > 0 else None

    def _get_nbr_tilt_idx(self, y_idx):
        # ignore the same bucket.
        nbr_y_idx = self.neighbour_tilt_idx(y_idx)
        if nbr_y_idx is not None:
            return nbr_y_idx

        direction = np.random.choice([-1, 1])
        iteration = 0
        while nbr_y_idx is None:
            iteration += 1
            if iteration > len(self._yaw):
                print('Yaw inf loop', y_idx, nbr_y_idx)
                return None
            new_idx = (y_idx + direction * iteration) % len(self._yaw)
            nbr_y_idx = self.neighbour_tilt_idx(new_idx, direction=direction, ignore_count=0, neighbour_count=1)

        return nbr_y_idx

    def _get_nbr_len_idx(self, p_idx, nbr_y_idx):
        nbr_p_idx = self.neighbour_len_idx(p_idx, nbr_y_idx)

        if nbr_p_idx is not None:
            return nbr_p_idx

        iteration = 0
        direction = np.random.choice([-1, 1])
        while nbr_p_idx is None:
            iteration += 1
            if iteration == len(self._pitch) + 1:
                # It needs to be done. Say the buckets are [0,35], In direction 1, 0 bucket will never come.
                direction = -1 * direction

            if iteration > 2 * len(self._pitch):
                print('Pitch inf loop', p_idx, nbr_p_idx, nbr_y_idx)
                return None

            new_idx = (p_idx + direction * iteration) % len(self._pitch)
            nbr_p_idx = self.neighbour_len_idx(new_idx,
                                               nbr_y_idx,
                                               direction=direction,
                                               ignore_count=0,
                                               neighbour_count=1)
        return nbr_p_idx

    def __getitem__(self, idx):
        y_idx, p_idx = self.idxdata[idx]
        if y_idx is None or p_idx is None:
            return None

        if self._use_same_bucket:
            nbr_idx = idx
            while nbr_idx == idx:
                nbr_idx = self.leveldata[y_idx][p_idx].get_random()
                if len(self.leveldata[y_idx][p_idx]) == 1:
                    print('Single entry', y_idx, p_idx)
                    return idx

            return nbr_idx

        nbr_y_idx = self._get_nbr_tilt_idx(y_idx)
        nbr_p_idx = self._get_nbr_len_idx(p_idx, nbr_y_idx)
        assert nbr_p_idx is not None, f'{y_idx}=> {nbr_y_idx}, {p_idx} => None'
        # print('Id wise', self.idxdata[idx], (nbr_y_idx, nbr_p_idx))
        nbr_idx = self.leveldata[nbr_y_idx][nbr_p_idx].get_random()
        return nbr_idx


# if __name__ == '__main__':
#     import pickle

#     def toRad(angle):
#         return math.pi * angle / 180

#     class Dummy:
#         def __init__(self):
#             self._len = 20
#             self.data = [(None, self.sample_gaze()) for _ in range(self._len)]

#         def sample_gaze(self):
#             yaw = math.pi * 2 * (np.random.rand() - 0.5)
#             pitch = math.pi * (np.random.rand() - 0.5)
#             return (yaw, pitch)

#         def get_gaze(self, idx):
#             return self.data[idx][1]

#         def __len__(self):
#             return self._len

#     dataset = Dummy()
#     # if os.path.exists('dummy.pkl'):
#     #     with open('dummy.pkl', 'rb') as f:
#     #         dataset = pickle.load(f)
#     #         import pdb
#     #         pdb.set_trace()
#     # else:
#     #     dataset = Dummy()
#     #     with open('dummy.pkl', 'wb') as f:
#     #         pickle.dump(dataset, f)

#     yaw_bin_size = 5
#     pitch_bin_size = 5
#     yaw_levels = Levels(toRad(yaw_bin_size), toRad(180), toRad(-180))
#     pitch_levels = Levels(toRad(pitch_bin_size), toRad(90), toRad(-90))
#     pair_sampler = DatasetIndexSampler(dataset, yaw_levels, pitch_levels, sample_nbr_cnt=1)
#     pair_sampler.create()

#     toDegree = 180 / math.pi
#     for idx in tqdm(np.random.randint(0, high=len(dataset) - 1, size=100)):
#         nbr_idx = pair_sampler[idx]
#         gaze = dataset.get_gaze(idx)
#         nbr_gaze = dataset.get_gaze(nbr_idx)
#         # print((toDegree * np.array(gaze)).round(1), '\t', (toDegree * np.array(nbr_gaze)).round(1))
