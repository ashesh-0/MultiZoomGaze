import h5py
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


class MPIIGazeImagerLoader(data.Dataset):
    """
    Working with normalized MPIIFaceGaze dataset.
    """

    def __init__(self, person_fpaths, transform=None):
        self._fpaths = person_fpaths
        self._handles = [h5py.File(fpath, 'r') for fpath in self._fpaths]
        self._size = 3000
        self._transform = transform

    def __getitem__(self, index):
        p_idx = index // self._size
        assert p_idx < len(self._handles), f'Invalid index:{index}. Maximum allowed {len(self._handles)*self._size -1}'

        idx = index % self._size
        img = np.array(self._handles[p_idx]['Data']['data'][idx], dtype=np.uint8)
        # RGB as the last dimension.
        img = np.moveaxis(img, 0, -1)
        img = Image.fromarray(img)

        # We want Yaw to be the first angle and pitch to be second. This convention is used in
        # Gaze360. So, here we are reversing the array.
        gaze = np.flip(self._handles[p_idx]['Data']['label'][idx, :2]).astype(np.float32)
        if self._transform is not None:
            img = self._transform(img)

        gaze = torch.FloatTensor(gaze)
        return (img, gaze)

    def __len__(self):
        return self._size * len(self._handles)
