import os

import torch
from torchvision import transforms

from core.constant import LIGHTNING_CHECKPOINT_DIR
from core.train_utils import checkpoint_fname_noext
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule


class BaseLightningModule(LightningModule):
    def __init__(self, model_type, lr=1e-4, batch_size=128, workers=4):
        super().__init__()
        self._model = None
        self._criterion = None
        self._lr = lr
        self._batch_size = batch_size
        self._workers = workers
        self._model_type = model_type
        self._train_fpath = 'train.txt'
        self._val_fpath = 'validation.txt'
        self._image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def file_startswith(self, folder, prefix):
        return len([filename for filename in os.listdir(folder) if filename.startswith(prefix)])

    def add_version(self, static):
        prefix = self.checkpoint_fpath(static) + '_epoch='
        pref_cnt = self.file_startswith(LIGHTNING_CHECKPOINT_DIR, prefix)
        if pref_cnt > 0:
            self.hparams['v'] = pref_cnt

    def get_checkpoint_callback(self, static):
        self.add_version(static)
        return ModelCheckpoint(
            filepath=os.path.join(LIGHTNING_CHECKPOINT_DIR, '_{epoch}_{val_loss:.3f}'),
            save_top_k=True,
            verbose=True,
            monitor='val_loss',
            mode='min',
            prefix=self.checkpoint_fpath(static))

    def checkpoint_fpath(self, static):
        return checkpoint_fname_noext(
            static,
            -1,
            [('TYPE', self._model_type)] + sorted([(k, v) for k, v in self.hparams.items()]),
            equalto_str='=',
        )

    def forward(self, x):
        return self._model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self._criterion(y_hat, y)
        tensorboard_log = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_log}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self._criterion(y_hat, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_log = {'val_loss': val_loss_mean}
        return {'val_loss': val_loss_mean, 'log': tensorboard_log}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._lr)
