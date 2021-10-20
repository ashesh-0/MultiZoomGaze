import torch.utils.data as data

from data_loader import default_loader
from RTGENE.data_utils import DataType, create_dataset


class RTGeneImagerLoaderStaticModel(data.Dataset):
    """
    Returns face images from RTGENE
    """

    def __init__(
            self,
            directory,
            train_file,
            transform,
            person_list=None,
            data_type=DataType.OriginalLarge,
            loader=default_loader,
    ):
        super().__init__()
        assert train_file is None, 'It is added just for making it similar to other data loader'
        self._dir = directory
        self._person_list = person_list
        self._loader = loader
        self._imgs = create_dataset(directory, person_list, data_type=data_type)
        self._transform = transform
        print(f'[{self.__class__.__name__}] person_list:{person_list}')

    def __getitem__(self, index):
        fpath, gaze = self._imgs[index]
        img = self._loader(fpath)
        img = self._transform(img)
        return img, gaze

    def __len__(self):
        return len(self._imgs)
