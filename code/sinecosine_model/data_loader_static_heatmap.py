from typing import Dict, List

import numpy as np
import pandas as pd

from core.pickle_utils import loadPickle
from sinecosine_model.data_loader_static_sinecosine import ImageLoaderStaticSineCosine


def get_personid_dict(csv_path, personid_pkl_path):
    df = pd.read_csv(csv_path, index_col=0)
    target_df = df[['fpath']].copy()
    person_id_dict = loadPickle(personid_pkl_path)

    def get_person_id(fpath):
        tokens = fpath.split('/')
        pid = tokens[-2]
        sid = tokens[-4]
        return person_id_dict[sid][pid]

    target_df['PersonId'] = target_df['fpath'].map(get_person_id)
    return target_df.set_index('fpath')['PersonId'].to_dict()


def get_target_dict(csv_path, centercrop_cols):
    df = pd.read_csv(csv_path, index_col=0)
    Y = np.argmin(df[centercrop_cols].values, axis=1)
    target_df = df[['fpath']].copy()
    target_df['Y'] = Y
    return target_df.set_index('fpath')['Y'].to_dict()


class ImageLoaderStaticHeatmap(ImageLoaderStaticSineCosine):
    def __init__(
            self,
            target_dict: Dict[str, int],
            personId_dict: Dict[str, int],
            allowed_personIds: List[int],
            source_path: str,
            file_name=None,
            transform=None,
    ):
        super().__init__(source_path, file_name=file_name, transform=transform)
        self._target_dict = target_dict
        self._personId_dict = personId_dict
        self._allowed_personIds = allowed_personIds
        if self._allowed_personIds is not None:
            imgs = []
            for fpath, gaze in self.imgs:
                if self._personId_dict[fpath] in self._allowed_personIds:
                    imgs.append((fpath, gaze))
            print(f'[{self.__class__.__name__}] Retained image percentage:{round(100*len(imgs)/len(self.imgs),2)}')
            self.imgs = imgs
        else:
            print(f'[{self.__class__.__name__}] Using all images')

    def __getitem__(self, index):
        source_img, _ = super().__getitem__(index)
        return source_img, self._target_dict[self.imgs[index][0]]
