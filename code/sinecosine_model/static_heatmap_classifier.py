import torch
import torch.nn as nn

from sinecosine_model.static_sinecosine_model import GazeStaticSineAndCosineModel


class StaticHeatmapClassifier(nn.Module):
    def __init__(self, num_classes, checkpoint_fpath, model_args=[], model_kwargs={}):
        super().__init__()
        self._N = num_classes
        self._fpath = checkpoint_fpath
        model_v = GazeStaticSineAndCosineModel(*model_args, **model_kwargs)
        model = torch.nn.DataParallel(model_v).cuda()
        _ = model.cuda()

        checkpoint = torch.load(checkpoint_fpath)
        model.load_state_dict(checkpoint['state_dict'])

        # self._model = model_v.base_model
        # self.final_layer = nn.Linear(model_kwargs['fc2'], self._N)

        layers = list(model_v.base_model.children())[:4]
        # layers += [nn.AdaptiveAvgPool2d((1, 1))]
        self._model = nn.Sequential(*layers)
        self.final_layer = nn.Sequential(nn.Linear(56 * 56 * 2, 64), nn.ReLU(), nn.Linear(64, self._N))

        for param in self._model.parameters():
            param.requires_grad = False

        print(f'[{self.__class__.__name__}] Loaded from {self._fpath}')

    def forward(self, img):
        fmap = self._model(img)
        fmap_std = fmap.std([1])
        fmap_mean = fmap.mean([1])
        feature = torch.cat([fmap_std, fmap_mean], dim=1)
        feature = feature.view((img.shape[0], -1))
        return self.final_layer(feature)
