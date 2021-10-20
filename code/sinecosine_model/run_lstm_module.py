import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from core.base_lightning_module import BaseLightningModule
from core.constant import IMG_DIR
from core.model_type import ModelType
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sinecosine_model.data_loader_sinecosine import ImageLoaderSineCosine, ImageLoaderSineCosineMultiSizedCrops
from sinecosine_model.mixed_loss import RegularizedSinAndCosLoss
from sinecosine_model.model import GazeSinCosLSTM
from sinecosine_model.model_with_lstm_scaling import GazeSinCosLSTMLstmScaling
from sinecosine_model.train_utils import compute_angular_error_sine_and_cosine


class SineCosineLstmModule(BaseLightningModule):
    def __init__(
            self,
            model_type,
            fc2=256,
            spatial_transformer=False,
            post_stn_croplist=None,
            cropsize_list=None,
            scaling_dict=None,
            lr=1e-4,
            batch_size=128,
            seq_len=7,
            workers=4,
    ):
        super().__init__(model_type, lr=lr, batch_size=batch_size, workers=workers)
        assert model_type in [ModelType.SinCosRegModel, ModelType.SinCosRegLstmScaleModel]

        self._cropsize_list = cropsize_list
        self._seq_len = seq_len
        if model_type == ModelType.SinCosRegModel:
            self._model = GazeSinCosLSTM(
                fc2=fc2, spatial_transformer=spatial_transformer, post_stn_croplist=post_stn_croplist)
        elif model_type == ModelType.SinCosRegLstmScaleModel:
            assert isinstance(scaling_dict, dict)
            self._model = GazeSinCosLSTMLstmScaling(self._seq_len, scaling_dict, fc2=fc2)

        self._criterion = RegularizedSinAndCosLoss()
        self.hparams = {
            'fc2': fc2,
            'batch': self._batch_size,
            'lr': self._lr,
        }
        if spatial_transformer:
            self.hparams['stn'] = spatial_transformer
        if post_stn_croplist is not None:
            self.hparams['stn_list'] = post_stn_croplist
        if cropsize_list is not None:
            self.hparams['c_list'] = cropsize_list
        if scaling_dict is not None:
            print('Scaling dict:', scaling_dict)
            min_scale = np.min([np.min(scaling_dict[k]) for k in scaling_dict])
            max_scale = np.max([np.max(scaling_dict[k]) for k in scaling_dict])
            std_scale = int(np.std([np.mean(scaling_dict[k]) for k in scaling_dict]))
            self.hparams['c_list'] = f'{min_scale}-{max_scale}-{std_scale}'

    def train_dataloader(self):
        train_img_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self._image_normalize,
        ])

        if self._cropsize_list is None:
            dataset = ImageLoaderSineCosine(
                IMG_DIR, self._train_fpath, transform=train_img_transforms, seq_len=self._seq_len)
        else:
            dataset = ImageLoaderSineCosineMultiSizedCrops(
                IMG_DIR,
                self._train_fpath,
                transform=None,
                cropsize_list=self._cropsize_list,
                seq_len=self._seq_len,
            )

        print('Total training imgs', len(dataset))
        loader = DataLoader(dataset, batch_size=self._batch_size, num_workers=self._workers, shuffle=True)
        return loader

    def val_dataloader(self):
        val_img_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self._image_normalize,
        ])

        # dataset = ImageLoaderStaticSineCosineMultiCropImgs(IMG_DIR, self._val_fpath, val_img_transforms, self._c_list)
        if self._cropsize_list is None:
            dataset = ImageLoaderSineCosine(
                IMG_DIR,
                self._val_fpath,
                transform=val_img_transforms,
                seq_len=self._seq_len,
            )
        else:
            dataset = ImageLoaderSineCosineMultiSizedCrops(
                IMG_DIR,
                self._val_fpath,
                transform=val_img_transforms,
                cropsize_list=self._cropsize_list,
                seq_len=self._seq_len,
            )

        print('Total validation imgs', len(dataset))
        loader = DataLoader(dataset, batch_size=self._batch_size, num_workers=self._workers, shuffle=False)
        return loader

    def training_step(self, batch, batch_idx):
        x, y = batch
        # y = batch[-1]
        y_hat, var = self(x)
        loss = self._criterion(y_hat, y, var)
        angular = compute_angular_error_sine_and_cosine(y_hat, y)
        tensorboard_log = {'train_loss': loss, 'train_angular': angular}
        tqdm_dict = {'train_loss': round(loss.item(), 3), 'angular': angular}

        return {'loss': loss, 'progress_bar': tqdm_dict, 'log': tensorboard_log}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, var = self(x)
        loss = self._criterion(y_hat, y, var)
        angular = compute_angular_error_sine_and_cosine(y_hat, y)
        tensorboard_log = {'val_loss': loss, 'val_angular': angular}
        return {'val_loss': loss, 'val_angular': angular, 'log': tensorboard_log}

    def validation_epoch_end(self, outputs):
        val_angular_mean = 0
        for output in outputs:
            val_angular_mean += output['val_angular']

        val_angular_mean /= len(outputs)
        tqdm_dict = {'val_angular': val_angular_mean}

        # show val_angular in progress bar but only log val_loss
        # results = {'progress_bar': tqdm_dict, 'log': {'val_angular': val_angular_mean.item()}}
        return tqdm_dict

    def get_checkpoint_callback(self, static):
        self.add_version(static)
        return ModelCheckpoint(
            filepath='/home/ashesh/checkpoints/_{epoch}_{val_angular:.3f}',
            save_top_k=True,
            verbose=True,
            monitor='val_angular',
            mode='min',
            prefix=self.checkpoint_fpath(static))


def main():
    batch_size = 32
    # crop_list = [223, 150]
    # fc2 = 256
    # spatial_transformer = True
    # post_stn_croplist = None
    # cropsize_list = [224, 220, 175, 150, 175, 200, 224]

    # model = SineCosineLstmModule(
    #     ModelType.SinCosRegModel,
    #     fc2=fc2,
    #     spatial_transformer=spatial_transformer,
    #     post_stn_croplist=post_stn_croplist,
    #     cropsize_list=cropsize_list,
    #     batch_size=batch_size,
    # )
    fc2 = 256
    seq_len = 5
    scaling_dict = {
        0: [224, 175, 125],
        1: [224, 175, 125],
        2: [224, 175, 125],
        3: [224, 175, 125],
        4: [224, 175, 125],
        # 5: [224, 150],
        # 6: [224, 150]
    }
    model = SineCosineLstmModule(
        ModelType.SinCosRegLstmScaleModel,
        scaling_dict=scaling_dict,
        fc2=fc2,
        batch_size=batch_size,
        seq_len=seq_len,
        workers=6)

    trainer = Trainer(
        gpus=2,
        max_epochs=30,
        checkpoint_callback=model.get_checkpoint_callback(False),
        precision=16,
        batch_size=batch_size,
        # profiler=True,
        # train_percent_check=0.01,
        # val_percent_check=0.1,
    )
    trainer.fit(model)


if __name__ == '__main__':
    main()
