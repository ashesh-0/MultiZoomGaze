import math
import os
import pickle

import numpy as np
from tqdm import tqdm


class OneLevel:
    def __init__(self, prop1_range, prop2_range):
        self.prop1 = prop1_range
        self.prop2 = prop2_range
        self.data = []

    def add(self, idx):
        self.data.append(idx)

    def get_random(self):
        return np.random.choice(self.data)

    def __len__(self):
        return len(self.data)


class Levels:
    def __init__(self, bin_size, max_val, min_val, levels=None):
        self.max = max_val
        self.min = min_val
        self.bin_size = bin_size
        if levels is None:
            self.levels = list(np.arange(self.min, self.max, self.bin_size))
            self.levels.append(self.max)
        else:
            self.levels = levels
        # print([round(x / math.pi * 180, 1) for x in self.levels])

    def __len__(self):
        return len(self.levels) - 1

    def __getitem__(self, idx):
        return self.levels[idx], self.levels[idx + 1]

    def get_idx(self, value):
        assert value >= self.levels[0]
        for i in range(len(self)):
            if value >= self.levels[i] and value < self.levels[i + 1]:
                return i

        raise Exception(f'Out of range value:{value}. It caters to {self.min} to {self.max}')


class DataSource:
    Gaze360 = 0
    XGaze = 1


class DatasetIndexSampler:
    def __init__(
        self,
        dataset,
        yaw_levels,
        pitch_levels,
        ignore_nbr_cnt=0,
        sample_nbr_cnt=1,
        ignore_same_bucket=True,
        data_source=DataSource.XGaze,
    ):
        """

        We have dividied the yaw and pitch angles into discrete bins of sizes yaw_bin_size and pitch_bin_size resp.
        Given a point in a bin B,
            a. If ignore_same_bucket is False then we sample another point from same bin.
            b. If ignore_same_bucket is True, then we sample another point from a neighbouring bin.
                * If We ignore ignore_nbr_cnt number of bins on either side of the original bin.
                * We sample a bin randomly from next sample_nbr_cnt bins on either side of the original bin B.
        Args:
            ignore_nbr_cnt: Ignore this many number of buckets in the neightbourhood on both sides.
            sample_nbr_cnt: After ignore region, sample from these many regions.
            ignore_same_bucket: Whether to ignore same bucket or not.
        """
        self._dataset = dataset
        self._yaw = yaw_levels
        self._pitch = pitch_levels
        self.leveldata = {}
        self.idxdata = []
        self._ign_same = ignore_same_bucket
        self._ign_cnt = ignore_nbr_cnt
        self._sample_nbr_cnt = sample_nbr_cnt
        self._data_source = data_source

    def pickle_fpath(self):
        toDeg = 180 / math.pi
        src = 'XG' if self._data_source == DataSource.XGaze else 'Gz360'
        path = f'DISampler_{src}_{len(self._dataset)}_Y{toDeg*self._yaw.bin_size}_P{toDeg*self._pitch.bin_size}.pkl'
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

    def create(self):
        if os.path.exists(self.pickle_fpath()):
            print(f'[{self.__class__.__name__}] Loading from {self.pickle_fpath()}')
            self.load()
            return
        for i in tqdm(range(len(self._dataset))):
            gaze = self._dataset.get_gaze(i)
            yaw, pitch = gaze
            y_idx = self._yaw.get_idx(yaw)
            p_idx = self._pitch.get_idx(pitch)

            self.idxdata.append((y_idx, p_idx))

            if y_idx not in self.leveldata:
                self.leveldata[y_idx] = {}
            if p_idx not in self.leveldata[y_idx]:
                y_range = self._yaw[y_idx]
                p_range = self._pitch[p_idx]
                self.leveldata[y_idx][p_idx] = OneLevel(y_range, p_range)

            self.leveldata[y_idx][p_idx].add(i)
        self.save()

    def nonempty_yaw_idx(self, idx):
        if idx not in self.leveldata or not self.leveldata[idx]:
            return False
        return True

    def nonempty_pitch_idx(self, idx, yaw_idx):
        return idx in self.leveldata[yaw_idx]

    def next_yaw_idx(self, idx):
        N = len(self._yaw)
        for next_idx in range(idx + 1, N):
            if self.nonempty_yaw_idx(next_idx):
                return next_idx
        return None

    def previous_yaw_idx(self, idx):
        for prev_idx in range(idx - 1, -1, -1):
            if self.nonempty_yaw_idx(prev_idx):
                return prev_idx
        return None

    def next_pitch_idx(self, idx, yaw_idx):
        N = len(self._pitch)
        for next_idx in range(idx + 1, N):
            if self.nonempty_pitch_idx(next_idx, yaw_idx):
                return next_idx

        return None

    def previous_pitch_idx(self, idx, yaw_idx):
        for prev_idx in range(idx - 1, -1, -1):
            if self.nonempty_pitch_idx(prev_idx, yaw_idx):
                return prev_idx
        return None

    def neighbour_yaw_idx(self, idx, direction=0, ignore_count=None, neighbour_count=None):
        assert self._ign_same is True
        if ignore_count is None:
            ignore_count = self._ign_cnt
            assert neighbour_count is None
            neighbour_count = self._sample_nbr_cnt

        diff_start = 1 + ignore_count
        diff_end = diff_start + neighbour_count
        idx_list = []
        if direction >= 0:
            n_idx = self.next_yaw_idx(idx + diff_start - 1)
            while n_idx is not None and n_idx < idx + diff_end:
                idx_list.append(n_idx)
                n_idx = self.next_yaw_idx(n_idx)

        if direction <= 0:
            p_idx = self.previous_yaw_idx(idx - diff_start + 1)
            while p_idx is not None and p_idx > idx - diff_end:
                idx_list.append(p_idx)
                p_idx = self.previous_yaw_idx(p_idx)

        return np.random.choice(idx_list) if len(idx_list) > 0 else None

    def neighbour_pitch_idx(self, idx, yaw_idx, direction=0, ignore_count=None, neighbour_count=None):
        assert self._ign_same is True
        if ignore_count is None:
            ignore_count = self._ign_cnt
            assert neighbour_count is None
            neighbour_count = self._sample_nbr_cnt

        diff_start = 1 + ignore_count
        diff_end = diff_start + neighbour_count
        idx_list = []
        if direction >= 0:
            n_idx = self.next_pitch_idx(idx + diff_start - 1, yaw_idx)
            while n_idx is not None and n_idx < idx + diff_end:
                idx_list.append(n_idx)
                n_idx = self.next_pitch_idx(n_idx, yaw_idx)

        if direction <= 0:
            p_idx = self.previous_pitch_idx(idx - diff_start + 1, yaw_idx)
            while p_idx is not None and p_idx > idx - diff_end:
                idx_list.append(p_idx)
                p_idx = self.previous_pitch_idx(p_idx, yaw_idx)

        return np.random.choice(idx_list) if len(idx_list) > 0 else None

    def _get_nbr_yaw_idx(self, y_idx):
        # ignore the same bucket.
        nbr_y_idx = self.neighbour_yaw_idx(y_idx)
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
            nbr_y_idx = self.neighbour_yaw_idx(new_idx, direction=direction, ignore_count=0, neighbour_count=1)

        return nbr_y_idx

    def _get_nbr_pitch_idx(self, p_idx, nbr_y_idx):
        nbr_p_idx = self.neighbour_pitch_idx(p_idx, nbr_y_idx)

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
            nbr_p_idx = self.neighbour_pitch_idx(new_idx,
                                                 nbr_y_idx,
                                                 direction=direction,
                                                 ignore_count=0,
                                                 neighbour_count=1)
        return nbr_p_idx

    def __getitem__(self, idx):
        y_idx, p_idx = self.idxdata[idx]
        if not self._ign_same:
            nbr_idx = idx
            while nbr_idx == idx:
                nbr_idx = self.leveldata[y_idx][p_idx].get_random()
                if len(self.leveldata[y_idx][p_idx]) == 1:
                    print('Single entry', y_idx, p_idx)
                    return idx

            return nbr_idx

        nbr_y_idx = self._get_nbr_yaw_idx(y_idx)
        nbr_p_idx = self._get_nbr_pitch_idx(p_idx, nbr_y_idx)
        assert nbr_p_idx is not None, f'{y_idx}=> {nbr_y_idx}, {p_idx} => None'
        # print('Id wise', self.idxdata[idx], (nbr_y_idx, nbr_p_idx))
        nbr_idx = self.leveldata[nbr_y_idx][nbr_p_idx].get_random()
        return nbr_idx


if __name__ == '__main__':
    import pickle

    def toRad(angle):
        return math.pi * angle / 180

    class Dummy:
        def __init__(self):
            self._len = 20
            self.data = [(None, self.sample_gaze()) for _ in range(self._len)]

        def sample_gaze(self):
            yaw = math.pi * 2 * (np.random.rand() - 0.5)
            pitch = math.pi * (np.random.rand() - 0.5)
            return (yaw, pitch)

        def get_gaze(self, idx):
            return self.data[idx][1]

        def __len__(self):
            return self._len

    dataset = Dummy()
    # if os.path.exists('dummy.pkl'):
    #     with open('dummy.pkl', 'rb') as f:
    #         dataset = pickle.load(f)
    #         import pdb
    #         pdb.set_trace()
    # else:
    #     dataset = Dummy()
    #     with open('dummy.pkl', 'wb') as f:
    #         pickle.dump(dataset, f)

    yaw_bin_size = 5
    pitch_bin_size = 5
    yaw_levels = Levels(toRad(yaw_bin_size), toRad(180), toRad(-180))
    pitch_levels = Levels(toRad(pitch_bin_size), toRad(90), toRad(-90))
    pair_sampler = DatasetIndexSampler(dataset, yaw_levels, pitch_levels, sample_nbr_cnt=1)
    pair_sampler.create()

    toDegree = 180 / math.pi
    for idx in tqdm(np.random.randint(0, high=len(dataset) - 1, size=100)):
        nbr_idx = pair_sampler[idx]
        gaze = dataset.get_gaze(idx)
        nbr_gaze = dataset.get_gaze(nbr_idx)
        # print((toDegree * np.array(gaze)).round(1), '\t', (toDegree * np.array(nbr_gaze)).round(1))
