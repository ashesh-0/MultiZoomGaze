import torch
import torch.utils.data as data

from data_loader import default_loader
from RTGENE.data_utils import DataType, create_dataset
from sinecosine_model.data_loader_sinecosine import centercrop_transform


class RTGeneImageLoaderLstmModel(data.Dataset):
    """
    Returns face images after inpainting
    """

    def __init__(
            self,
            directory,
            train_file,
            transform,
            person_list=None,
            cropsize_list=None,
            data_type=DataType.OriginalLarge,
            loader=default_loader,
            img_size=224,
    ):
        super().__init__()
        assert train_file is None, 'It is added just for making it similar to other data loader'
        assert transform is None, 'It is added just for making it similar to other data loader'
        self._img_size = img_size
        self._dir = directory
        self._person_list = person_list
        self._loader = loader
        self._cropsizes = cropsize_list
        self._imgs = create_dataset(directory, person_list, data_type=data_type)

        self._transforms = [centercrop_transform(c, self._img_size) for c in self._cropsizes]
        print(f'[{self.__class__.__name__}] person_list:{person_list} cropsizes:{self._cropsizes}')

    def __getitem__(self, index):
        fpath, gaze = self._imgs[index]
        video = self._get_video(fpath)
        return video, gaze

    def _get_video(self, fpath: str):
        seq_len = len(self._transforms)
        source_video = torch.FloatTensor(seq_len, 3, self._img_size, self._img_size)
        img = self._loader(fpath)
        for i, transform in enumerate(self._transforms):
            source_video[i, ...] = transform(img)

        source_video = source_video.view(3 * seq_len, self._img_size, self._img_size)
        return source_video

    def __len__(self):
        return len(self._imgs)
