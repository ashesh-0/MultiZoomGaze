import torch
import torch.utils.data as data

from core.dict_img_loader import DictEyeImgLoader
from core.gaze_utils import get_yaw_pitch
from data_loader_static_model import make_dataset_gaze360
from eye_model.data_loader_static_sinecosine import remove_eyeless_imgs
from sinecosine_model.train_utils import get_sincos_gaze_from_spherical


class ImageLoaderStaticSineCosineWithEyes(data.Dataset):
    def __init__(
            self,
            source_path,
            file_name=None,
            transform=None,
            g_z_min_val=None,
    ):

        loader_obj = DictEyeImgLoader(ignore_unavailable=False)
        imgs = make_dataset_gaze360(source_path, file_name, g_z_min_val=g_z_min_val)
        imgs = remove_eyeless_imgs(imgs, loader_obj)

        self.source_path = source_path
        self.file_name = file_name

        self.imgs = imgs
        self.transform = transform
        self.loader = loader_obj.load

    def get_gaze_vector(self, gaze):
        spherical_vector = get_yaw_pitch(gaze)
        return get_sincos_gaze_from_spherical(spherical_vector)

    def __getitem__(self, index):
        path_source, gaze = self.imgs[index]
        dct = self.loader(path_source)
        face_img = self.transform(dct['img'])
        gaze_sinecos_vector = self.get_gaze_vector(gaze)
        return face_img, gaze_sinecos_vector

    def __len__(self):
        return len(self.imgs)
