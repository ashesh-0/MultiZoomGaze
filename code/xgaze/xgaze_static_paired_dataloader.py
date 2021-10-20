import math
from typing import List

from torch.utils.data import Dataset

from core.paired_datasampler import DatasetIndexSampler, DataSource, Levels
from xgaze.xgaze_static_dataloader import GazeDataset


def toRad(angle):
    return math.pi * angle / 180


class PairedDataLoader(Dataset):
    def __init__(self,
                 dataset_path: str,
                 keys_to_use: List[str] = None,
                 sub_folder='',
                 yaw_bin_size=5,
                 pitch_bin_size=5,
                 ignore_nbr_cnt=0,
                 sample_nbr_cnt=1,
                 ignore_same_bucket=True,
                 transform=None,
                 is_shuffle=True,
                 index_file=None,
                 sc_target=False,
                 is_load_label=True):
        super().__init__()
        self.dataset = GazeDataset(dataset_path,
                                   keys_to_use=keys_to_use,
                                   sub_folder=sub_folder,
                                   transform=transform,
                                   is_shuffle=is_shuffle,
                                   index_file=index_file,
                                   sc_target=sc_target,
                                   is_load_label=is_load_label)

        yaw_levels = Levels(toRad(yaw_bin_size), toRad(180), toRad(-180))
        pitch_levels = Levels(toRad(pitch_bin_size), toRad(90), toRad(-90))

        # yaw_levels = Levels(toRad(yaw_bin_size), toRad(158), toRad(-179))
        # pitch_levels = Levels(toRad(pitch_bin_size), toRad(88), toRad(-86))
        self.pair_sampler = DatasetIndexSampler(
            self.dataset,
            yaw_levels,
            pitch_levels,
            ignore_nbr_cnt=ignore_nbr_cnt,
            sample_nbr_cnt=sample_nbr_cnt,
            ignore_same_bucket=ignore_same_bucket,
            data_source=DataSource.XGaze,
        )
        self.pair_sampler.create()
        print(f'[{self.__class__.__name__}] Bin:{yaw_bin_size},{pitch_bin_size} IC:{ignore_nbr_cnt}'
              f' SC:{sample_nbr_cnt} IgnoreB{ignore_same_bucket}')

    def __getitem__(self, i):
        pair_idx = self.pair_sampler[i]
        return (*self.dataset[i], *self.dataset[pair_idx])

    def __len__(self):
        return len(self.dataset)
