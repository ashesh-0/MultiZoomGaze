import math

from torch.utils.data import Dataset

from core.paired_datasampler import DatasetIndexSampler, DataSource, Levels
from data_loader_static_model import ImagerLoaderStaticModel


def toRad(angle):
    return math.pi * angle / 180


class PairedDataLoaderGaze360(Dataset):
    def __init__(self,
                 source_path,
                 file_name=None,
                 transform=None,
                 yaw_bin_size=5,
                 pitch_bin_size=5,
                 ignore_nbr_cnt=0,
                 sample_nbr_cnt=1,
                 ignore_same_bucket=True):
        super().__init__()
        self.dataset = ImagerLoaderStaticModel(
            source_path,
            file_name=file_name,
            transform=transform,
        )

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
            data_source=DataSource.Gaze360,
        )
        self.pair_sampler.create()
        print(f'[{self.__class__.__name__}] Bin:{yaw_bin_size},{pitch_bin_size} IC:{ignore_nbr_cnt}'
              f' SC:{sample_nbr_cnt} IgnoreB{ignore_same_bucket}')

    def __getitem__(self, i):
        pair_idx = self.pair_sampler[i]
        return (*self.dataset[i], *self.dataset[pair_idx])

    def __len__(self):
        return len(self.dataset)
