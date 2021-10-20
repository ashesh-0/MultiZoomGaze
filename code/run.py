"""
This script can be used to train/evaluate different models described in MultiZoomGaze paper.
"""
import argparse
import os
import socket
import sys
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.utils.cpp_extension import CUDA_HOME

from backbones.backbone_type import BackboneType
from core.loss import PinBallLoss
from core.model_type import ModelType
from core.train_utils import (checkpoint_fname, compute_angular_error, compute_angular_error_xyz, save_checkpoint)
from data_loader import ImagerLoader
from data_loader_multicrop import ImageLoaderMultiSizedCrops
from data_loader_static_model import ImagerLoaderStaticModel
from model import GazeLSTM
from non_lstm_based_model import GazeMultiCropModel
from run_utils import evaluate, train, validate
from sinecosine_model.data_loader_sinecosine import (ImageLoaderSineCosine, ImageLoaderSineCosineMultiScale,
                                                     ImageLoaderSineCosineMultiSizedCrops,
                                                     ImageLoaderSineCosineMultiSizedRandomCrops)
from sinecosine_model.data_loader_static_sinecosine import (ImageLoaderStaticMultiSineCosine,
                                                            ImageLoaderStaticSineCosine,
                                                            ImageLoaderStaticSineCosineMultiCenterCrops)
from sinecosine_model.lazy_aggregation_model import LazyAggregationModel
from sinecosine_model.mixed_loss import (RegularizedMultiSinAndCosLoss, RegularizedSinAndCosLoss, WeightedMseLoss,
                                         WeightedRegularizedSinAndCosLoss)
from sinecosine_model.model import GazeSinCosLSTM
from sinecosine_model.model_with_lstm_scaling import GazeSinCosLSTMLstmScalingV2
from sinecosine_model.non_lstm_based_model import AggregationType, GazeSinCosMultiCropModel
from sinecosine_model.static_sinecosine_model import GazeStaticMultiSineAndCosineModel, GazeStaticSineAndCosineModel
from sinecosine_model.train_utils import compute_angular_error_sine_and_cosine
from static_model import GazeStaticModel
from static_xyz.data_loader_static_xyz import ImagerLoaderStaticXyzModel
from static_xyz.static_xyz_model import GazeStaticXyzModel

WORKERS = 4
EPOCHS = 100
BATCH_SIZE = 64
BEST_ERROR = 100
LEARNING_RATE = 1e-4


def main(model_type, train_file, val_file, test_file, source_path, checkpoints_path, img_size, kfold_train, **params):
    global EPOCHS
    sum_writer = None
    # deprecated feature

    print('Train:', train_file)
    print('Val:', val_file)
    global args, BEST_ERROR
    checkpoint_fpath = os.path.join(checkpoints_path, f'gaze360_model_{model_type}.pth.tar')
    image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_img_transforms = transforms.Compose(
        [transforms.Resize((img_size, img_size)),
         transforms.ToTensor(), image_normalize])
    val_img_transforms = transforms.Compose(
        [transforms.Resize((img_size, img_size)),
         transforms.ToTensor(), image_normalize])
    unfreeze_epoch = None
    train_img_loader_kwargs = {}
    val_img_loader_kwargs = {}

    if model_type == ModelType.LSTM:
        print('Using LSTM model')
        model_v = GazeLSTM()
        ImageLoaderClass = ImagerLoader
        compute_angular_error_fn = compute_angular_error
        criterion = PinBallLoss().cuda()

    elif model_type == ModelType.LSTMBackward:
        raise Exception('Tri model is removed')

    elif model_type == ModelType.StaticModel:
        print('Using Static model')
        backbone_type = params['backbone_type']
        checkpoint_fpath = checkpoint_fname(
            True,
            kfold_train,
            [
                ('TYPE', model_type),
                ('bkb', backbone_type),
                ('bsz', BATCH_SIZE),
                ('lr', LEARNING_RATE),
                ('v', f'master_{params["version"]}'),
            ],
            dirname=checkpoints_path,
        )
        print(checkpoint_fpath)

        model_v = GazeStaticModel(backbone_type=backbone_type)
        ImageLoaderClass = ImagerLoaderStaticModel
        compute_angular_error_fn = compute_angular_error
        criterion = PinBallLoss().cuda()

    elif model_type == ModelType.StaticBackwardModel:
        raise Exception('Tri model is removed')
    elif model_type == ModelType.StaticXyzModel:
        model_v = GazeStaticXyzModel()
        checkpoint_fpath = checkpoint_fname(
            True,
            kfold_train,
            [
                ('TYPE', model_type),
                ('bsz', BATCH_SIZE),
                ('lr', LEARNING_RATE),
                ('v', f'master_{params["version"]}'),
            ],
            dirname=checkpoints_path,
        )
        print(checkpoint_fpath)
        ImageLoaderClass = ImagerLoaderStaticXyzModel
        compute_angular_error_fn = compute_angular_error_xyz
        criterion = PinBallLoss().cuda()
    elif model_type == ModelType.StaticSinModel:
        raise Exception('StaticSinModel is removed')
    elif model_type == ModelType.StaticCosModel:
        raise Exception('StaticCosModel is removed')
    elif model_type == ModelType.StaticSinCosModel:
        print('Using static sin cos model')
        backbone_type = params['backbone_type']
        fc2 = 256
        centercrop = None
        checkpoint_fpath = checkpoint_fname(
            True,
            kfold_train,
            [
                ('TYPE', model_type),
                ('fc2', fc2),
                ('bkb', backbone_type),
                # ('centercrop', centercrop),
                ('bsz', BATCH_SIZE),
                ('lr', LEARNING_RATE),
                ('v', f'master_{params["version"]}'),
            ],
            dirname=checkpoints_path,
        )
        print(checkpoint_fpath)
        train_img_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            # transforms.CenterCrop(centercrop),
            # transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            image_normalize,
        ])
        val_img_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            # transforms.CenterCrop(centercrop),
            # transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            image_normalize,
        ])
        model_v = GazeStaticSineAndCosineModel(fc2=fc2, backbone_type=backbone_type)
        ImageLoaderClass = ImageLoaderStaticSineCosine
        compute_angular_error_fn = compute_angular_error_sine_and_cosine
        criterion = PinBallLoss().cuda()
    elif model_type == ModelType.StaticMultiSinCosRegModel:
        print('Using regularized multi sin cos static model')
        fc2 = 256
        centercrop = 175
        checkpoint_fpath = checkpoint_fname(
            True,
            kfold_train,
            [
                ('TYPE', model_type),
                ('fc2', fc2),
                ('centercrop', centercrop),
                ('bsz', BATCH_SIZE),
                ('lr', LEARNING_RATE),
                ('v', f'master_{params["version"]}'),
            ],
            dirname=checkpoints_path,
        )
        print(checkpoint_fpath)
        ImageLoaderClass = ImageLoaderStaticMultiSineCosine
        train_img_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(centercrop),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            image_normalize,
        ])
        val_img_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(centercrop),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            image_normalize,
        ])
        model_v = GazeStaticMultiSineAndCosineModel(fc2=fc2)

        compute_angular_error_fn = compute_angular_error_sine_and_cosine
        criterion = RegularizedMultiSinAndCosLoss([0, 1, 3, 4], [2]).cuda()
    elif model_type == ModelType.StaticSinCosRegModel:
        print('Using regularized sin cos static model')
        backbone_type = params['backbone_type']
        fc2 = 256
        fc1 = None
        freeze_layer_idx = 0
        unfreeze_epoch = 0
        # backbone_loader_checkpoint_fpath = "/home/ashesh/model_best_gaze360_TYPE:10_fc2:256_time:False_diff_crop:224-150_tar_idx:3_seq_len:7_bsz:64_lr:0.0001_v:master_3.pth.tar"
        backbone_loader_checkpoint_fpath = None
        centercrop = 200
        use_extended_head = False

        checkpoint_args = [
            ('TYPE', model_type),
            ('fc1', fc1),
            ('fc2', fc2),
            ('bkb', backbone_type),
        ]
        if centercrop is not None:
            checkpoint_args += [('centercrop', centercrop)]

        if freeze_layer_idx > 0:
            checkpoint_args += [('freeze', freeze_layer_idx)]
        if backbone_loader_checkpoint_fpath is not None:
            checkpoint_args += [('bkb_load', 1)]
        checkpoint_args += [
            ('imsz', img_size),
            ('bsz', BATCH_SIZE),
            ('lr', LEARNING_RATE),
            ('v', f'master_{params["version"]}'),
        ]

        checkpoint_fpath = checkpoint_fname(
            True,
            kfold_train,
            checkpoint_args,
            dirname=checkpoints_path,
        )
        print(checkpoint_fpath)
        if use_extended_head:
            ImageLoaderClass = ImageLoaderStaticSineCosineMultiCenterCrops
            train_img_transforms = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                image_normalize,
            ])
            val_img_transforms = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                image_normalize,
            ])
        else:
            ImageLoaderClass = ImageLoaderStaticSineCosine
            train_img_transforms = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.CenterCrop(centercrop),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                image_normalize,
            ])
            val_img_transforms = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.CenterCrop(centercrop),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                image_normalize,
            ])
        model_v = GazeStaticSineAndCosineModel(
            fc2=fc2,
            fc1=fc1,
            freeze_layer_idx=freeze_layer_idx,
            backbone_type=backbone_type,
            backbone_loader_checkpoint_fpath=backbone_loader_checkpoint_fpath,
        )

        compute_angular_error_fn = compute_angular_error_sine_and_cosine
        criterion = RegularizedSinAndCosLoss().cuda()
    elif model_type == ModelType.SinCosRegModel:
        print('Using regularized sin cos LSTM model')
        fc2 = 256
        backbone_type = params['backbone_type']
        freeze_layer_idx = 0
        unfreeze_epoch = 0

        assert params['bidirectional_lstm'] in [0, 1]
        bidirectional_lstm = params['bidirectional_lstm'] == 1

        assert isinstance(params['cropsize_list'], list) or params['cropsize_list'] is None
        cropsize_list = params['cropsize_list']  #[224, 200, 175, 150, 175, 200, 224]

        enable_time = params['enable_time']
        if params['symmetric'] == 1 and cropsize_list is not None:
            cropsize_list = cropsize_list + cropsize_list[:-1][::-1]

        seq_len = len(cropsize_list) if cropsize_list is not None else 7
        if bidirectional_lstm and params['symmetric']:
            target_seq_index = seq_len // 2
        else:
            target_seq_index = seq_len - 1

        ckp_tples = [
            ('TYPE', model_type),
            ('bkb', backbone_type),
            ('fc2', fc2),
            ('time', enable_time),
        ]
        if cropsize_list is not None:
            ckp_tples += [('diff_crop', f'{cropsize_list[0]}-{cropsize_list[seq_len//2]}')]

        ckp_tples += [
            ('tar_idx', target_seq_index),
            ('seq_len', seq_len),
            ('imsz', img_size),
            ('bsz', BATCH_SIZE),
            ('lr', LEARNING_RATE),
        ]
        if bidirectional_lstm is False:
            ckp_tples += [('bidir', int(bidirectional_lstm))]

        ckp_tples += [
            ('v', f'master_{params["version"]}'),
        ]
        checkpoint_fpath = checkpoint_fname(
            False,
            kfold_train,
            ckp_tples,
            dirname=checkpoints_path,
        )
        print(checkpoint_fpath)
        model_v = GazeSinCosLSTM(
            fc2=fc2,
            freeze_layer_idx=freeze_layer_idx,
            target_seq_index=target_seq_index,
            seq_len=seq_len,
            backbone_type=backbone_type,
            bidirectional_lstm=bidirectional_lstm,
        )
        train_img_loader_kwargs = {'enable_time': enable_time}
        val_img_loader_kwargs = {'enable_time': enable_time}
        if cropsize_list is None:
            ImageLoaderClass = ImageLoaderSineCosine
        else:
            ImageLoaderClass = ImageLoaderSineCosineMultiSizedCrops
            train_img_loader_kwargs['cropsize_list'] = cropsize_list
            train_img_loader_kwargs['seq_len'] = seq_len
            train_img_loader_kwargs['img_size'] = img_size

            val_img_loader_kwargs['cropsize_list'] = cropsize_list
            val_img_loader_kwargs['seq_len'] = seq_len
            val_img_loader_kwargs['img_size'] = img_size

        compute_angular_error_fn = compute_angular_error_sine_and_cosine
        criterion = RegularizedSinAndCosLoss().cuda()
    elif model_type == ModelType.SinCosModel:
        print('Using sin cos LSTM model')
        fc2 = 256
        enable_time = params['enable_time']
        cropsize_list = params['cropsize_list']
        seq_len = len(cropsize_list)
        if params['symmetric'] == 1:
            cropsize_list = cropsize_list + cropsize_list[:-1][::-1]
            target_seq_index = seq_len // 2
        else:
            target_seq_index = seq_len - 1

        checkpoint_fpath = checkpoint_fname(
            enable_time,
            kfold_train,
            [
                ('TYPE', model_type),
                ('fc2', fc2),
                ('time', enable_time),
                ('diff_crop', f'{cropsize_list[0]}-{cropsize_list[seq_len//2]}'),
                ('tar_idx', target_seq_index),
                ('seq_len', seq_len),
                ('imsz', img_size),
                ('bsz', BATCH_SIZE),
                ('lr', LEARNING_RATE),
                ('v', f'master_{params["version"]}'),
            ],
            dirname=checkpoints_path,
        )
        print(checkpoint_fpath)
        model_v = GazeSinCosLSTM(
            fc2=fc2,
            target_seq_index=target_seq_index,
            seq_len=seq_len,
        )
        train_img_loader_kwargs = {'enable_time': enable_time, 'seq_len': seq_len}
        val_img_loader_kwargs = {'enable_time': enable_time, 'seq_len': seq_len}
        if cropsize_list is None:
            ImageLoaderClass = ImageLoaderSineCosine
        else:
            ImageLoaderClass = ImageLoaderSineCosineMultiSizedCrops
            train_img_loader_kwargs['cropsize_list'] = cropsize_list
            val_img_loader_kwargs['cropsize_list'] = cropsize_list

        compute_angular_error_fn = compute_angular_error_sine_and_cosine
        criterion = PinBallLoss().cuda()
    elif model_type == ModelType.SinCosRegLstmScaleModel:
        img_size = 150
        fc2 = 256
        seq_len = 5
        cropsize_list = [224, 160, 100]
        checkpoint_fpath = checkpoint_fname(
            False,
            kfold_train,
            [
                ('TYPE', model_type),
                ('fc2', fc2),
                ('scale', f'{cropsize_list[0]}-{cropsize_list[-1]}-{len(cropsize_list)}'),
                ('T', seq_len),
                ('img_sz', img_size),
                ('bsz', BATCH_SIZE),
                ('lr', LEARNING_RATE),
            ],
            dirname=checkpoints_path,
        )
        print(checkpoint_fpath)
        model_v = GazeSinCosLSTMLstmScalingV2(seq_len, len(cropsize_list), fc2=fc2)
        ImageLoaderClass = ImageLoaderSineCosineMultiScale
        train_img_loader_kwargs = {'seq_len': seq_len, 'cropsize_list': cropsize_list, 'img_size': img_size}
        val_img_loader_kwargs = {'seq_len': seq_len, 'cropsize_list': cropsize_list, 'img_size': img_size}
        compute_angular_error_fn = compute_angular_error_sine_and_cosine
        criterion = RegularizedSinAndCosLoss().cuda()

    elif model_type == ModelType.StaticSinCosAllRegModel:
        raise Exception('StaticSineAndCosineAllModel is removed')

    elif model_type == ModelType.StaticWeightedSinCosRegModel:
        print('Using weighted regularized sin cos static model')
        w_lambda = 0.005
        checkpoint_fpath = checkpoint_fname(
            True,
            kfold_train,
            [
                ('TYPE', model_type),
                ('w_loss', w_lambda),
            ],
            dirname=checkpoints_path,
        )

        print(checkpoint_fpath)
        model_v = GazeStaticSineAndCosineModel()
        ImageLoaderClass = ImageLoaderStaticSineCosine
        compute_angular_error_fn = compute_angular_error_sine_and_cosine
        criterion = WeightedRegularizedSinAndCosLoss(w_lambda=w_lambda).cuda()
    elif model_type == ModelType.StaticWeightedMseSinCosModel:
        print('Using weighted mse sin cos static model')
        w_lambda = 0.1
        checkpoint_fpath = checkpoint_fname(
            True,
            kfold_train,
            [
                ('TYPE', model_type),
                ('w_loss', w_lambda),
            ],
            dirname=checkpoints_path,
        )

        print(checkpoint_fpath)
        model_v = GazeStaticSineAndCosineModel()
        ImageLoaderClass = ImageLoaderStaticSineCosine
        compute_angular_error_fn = compute_angular_error_sine_and_cosine
        criterion = WeightedMseLoss(w_lambda=w_lambda).cuda()

    elif model_type == ModelType.Gaze360MultiCropModel:
        print("Using MultiCrops with Gaze360 target and loss")
        backbone_type = BackboneType.Resnet18
        enable_time = params['enable_time']
        # cropsize_list = None
        # target_seq_index = None
        cropsize_list = [224, 200, 175, 150, 175, 200, 224]
        seq_len = len(cropsize_list)
        target_seq_index = 3

        checkpoint_fpath = checkpoint_fname(
            False,
            kfold_train,
            [
                ('TYPE', model_type),
                ('bkb', backbone_type),
                ('time', enable_time),
                ('diff_crop', f'{cropsize_list[0]}-{cropsize_list[seq_len//2]}'),
                ('tar_idx', target_seq_index),
                ('seq_len', seq_len),
                ('bsz', BATCH_SIZE),
                ('lr', LEARNING_RATE),
                ('v', f'master_{params["version"]}'),
            ],
            dirname=checkpoints_path,
        )
        print(checkpoint_fpath)
        model_v = GazeLSTM(
            output_dim=3,
            target_seq_index=target_seq_index,
            seq_len=seq_len,
            backbone_type=backbone_type,
        )
        train_img_loader_kwargs = {'enable_time': enable_time}
        val_img_loader_kwargs = {'enable_time': enable_time}
        ImageLoaderClass = ImageLoaderMultiSizedCrops
        train_img_loader_kwargs['cropsize_list'] = cropsize_list

        val_img_loader_kwargs['cropsize_list'] = cropsize_list

        compute_angular_error_fn = compute_angular_error
        criterion = PinBallLoss().cuda()
    elif model_type in [
            ModelType.NonLstmSinCosRegModel, ModelType.NonLstmSinCosModel, ModelType.NonLstmSinCosRandomModel
    ]:
        if model_type == ModelType.NonLstmSinCosModel:
            print('Using Non-LSTM based Sine Cosine model')
        else:
            print('Using Non-LSTM based Sine Cosine Regularized model')

        backbone_type = params['backbone_type']
        enable_time = params['enable_time']
        assert isinstance(params['cropsize_list'], list)
        cropsize_list = params['cropsize_list']

        assert params['symmetric'] == 0, 'symmetric=1 is not handled in this case. Pass full cropsize instead'

        seq_len = len(cropsize_list)
        atype = params['atype']
        if isinstance(params['dataloader_params'], dict):
            future_prediction = bool(params['dataloader_params'].get('future_prediction', 0))
        else:
            future_prediction = False

        if len(set(cropsize_list)) > 1:
            diff_crop_str = f'{cropsize_list[0]}-{cropsize_list[-1]}'
        else:
            diff_crop_str = f'One-{cropsize_list[0]}'
        ckp_tples = [
            ('TYPE', model_type),
            ('bkb', backbone_type),
            ('time', enable_time),
            ('diff_crop', diff_crop_str),
            ('seq_len', seq_len),
            ('atype', atype),
        ]
        if future_prediction:
            ckp_tples += [('fp', 1)]

        ckp_tples += [
            ('bsz', BATCH_SIZE),
            ('lr', LEARNING_RATE),
            ('v', f'master_{params["version"]}'),
        ]
        checkpoint_fpath = checkpoint_fname(
            False,
            kfold_train,
            ckp_tples,
            dirname=checkpoints_path,
        )
        print(checkpoint_fpath)
        model_v = GazeSinCosMultiCropModel(
            seq_len=seq_len,
            backbone_type=backbone_type,
            atype=atype,
            cropsize_list=cropsize_list,
        )

        train_img_loader_kwargs = {'enable_time': enable_time, 'future_prediction': future_prediction}
        val_img_loader_kwargs = {'enable_time': enable_time, 'future_prediction': future_prediction}
        if model_type == ModelType.NonLstmSinCosRandomModel:
            ImageLoaderClass = ImageLoaderSineCosineMultiSizedRandomCrops
        else:
            ImageLoaderClass = ImageLoaderSineCosineMultiSizedCrops
        train_img_loader_kwargs['cropsize_list'] = cropsize_list
        train_img_loader_kwargs['seq_len'] = seq_len
        train_img_loader_kwargs['img_size'] = img_size

        val_img_loader_kwargs['cropsize_list'] = cropsize_list
        val_img_loader_kwargs['seq_len'] = seq_len
        val_img_loader_kwargs['img_size'] = img_size

        compute_angular_error_fn = compute_angular_error_sine_and_cosine
        if model_type in [ModelType.NonLstmSinCosModel, ModelType.NonLstmSinCosRandomModel]:
            criterion = PinBallLoss().cuda()
        elif model_type == ModelType.NonLstmSinCosRegModel:
            criterion = RegularizedSinAndCosLoss().cuda()
        else:
            raise Exception(f'Unexpected model_type: {ModelType.name(model_type)}')

    elif model_type == ModelType.NonLstmMultiCropModel:
        print("Using Non lstm based MultiCrops with Gaze360 target and loss")
        backbone_type = params['backbone_type']
        enable_time = params['enable_time']
        cropsize_list = params['cropsize_list']
        seq_len = len(cropsize_list)
        assert params['symmetric'] == 0, 'symmetric=1 is not handled in this case. Pass full cropsize instead'

        atype = params['atype']

        checkpoint_fpath = checkpoint_fname(
            False,
            kfold_train,
            [
                ('TYPE', model_type),
                ('bkb', backbone_type),
                ('time', enable_time),
                ('diff_crop', f'{cropsize_list[0]}-{cropsize_list[seq_len//2]}'),
                ('seq_len', seq_len),
                ('atype', atype),
                ('bsz', BATCH_SIZE),
                ('lr', LEARNING_RATE),
                ('v', f'master_{params["version"]}'),
            ],
            dirname=checkpoints_path,
        )
        print(checkpoint_fpath)
        model_v = GazeMultiCropModel(
            output_dim=3,
            backbone_type=backbone_type,
            cropsize_list=cropsize_list,
            atype=atype,
        )
        train_img_loader_kwargs = {'enable_time': enable_time}
        val_img_loader_kwargs = {'enable_time': enable_time}
        ImageLoaderClass = ImageLoaderMultiSizedCrops
        train_img_loader_kwargs['cropsize_list'] = cropsize_list

        val_img_loader_kwargs['cropsize_list'] = cropsize_list

        compute_angular_error_fn = compute_angular_error
        criterion = PinBallLoss().cuda()

    elif model_type == ModelType.Gaze360LazyAggregationModel:
        print('Using Non-LSTM based lazy aggregation Sine Cosine model')

        backbone_type = params['backbone_type']
        assert isinstance(params['cropsize_list'], list)
        cropsize_list = params['cropsize_list']
        agg_layer_idx = params['agg_layer_idx']

        assert params['symmetric'] == 0, 'symmetric=1 is not handled in this case. Pass full cropsize instead'

        seq_len = len(cropsize_list)

        if len(set(cropsize_list)) > 1:
            diff_crop_str = f'{cropsize_list[0]}-{cropsize_list[-1]}'
        else:
            diff_crop_str = f'One-{cropsize_list[0]}'
        ckp_tples = [
            ('TYPE', model_type),
            ('bkb', backbone_type),
            ('diff_crop', diff_crop_str),
            ('seq_len', seq_len),
            ('agg_idx', agg_layer_idx),
            ('bsz', BATCH_SIZE),
            ('lr', LEARNING_RATE),
            ('v', f'master_{params["version"]}'),
        ]
        checkpoint_fpath = checkpoint_fname(
            False,
            kfold_train,
            ckp_tples,
            dirname=checkpoints_path,
        )
        print(checkpoint_fpath)
        model_v = LazyAggregationModel(agg_layer_idx=agg_layer_idx, backbone_type=backbone_type)
        ImageLoaderClass = ImageLoaderStaticSineCosine

        compute_angular_error_fn = compute_angular_error_sine_and_cosine
        criterion = PinBallLoss().cuda()

    else:
        raise Exception(f'Invalid model_type: {model_type}')

    model = torch.nn.DataParallel(model_v).cuda()
    model.cuda()

    cudnn.benchmark = True
    image_loader = ImageLoaderClass(source_path, train_file, train_img_transforms, **train_img_loader_kwargs)
    train_loader = torch.utils.data.DataLoader(image_loader,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=WORKERS,
                                               pin_memory=True)
    print('Train transforms', train_img_transforms)
    print('Val transforms', val_img_transforms)

    val_loader = torch.utils.data.DataLoader(ImageLoaderClass(source_path, val_file, val_img_transforms,
                                                              **val_img_loader_kwargs),
                                             batch_size=BATCH_SIZE,
                                             shuffle=True,
                                             num_workers=WORKERS,
                                             pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
    print('Overall Adam Optimizer')

    if params['evaluate']:
        print('Skipping Training')
        checkpoint_fpath = os.path.join(os.path.dirname(checkpoint_fpath),
                                        'model_best_' + os.path.basename(checkpoint_fpath))
        assert os.path.exists(checkpoint_fpath), checkpoint_fpath
        checkpoint = torch.load(checkpoint_fpath)
        model.load_state_dict(checkpoint['state_dict'])

        print(f"Loaded from {checkpoint_fpath}\n Epoch:{checkpoint['epoch']}")
        img_loader = ImageLoaderClass(source_path, test_file, val_img_transforms, **val_img_loader_kwargs)
        if evaluate(img_loader, model, criterion, compute_angular_error_fn, BATCH_SIZE, WORKERS):
            return

        test_loader = torch.utils.data.DataLoader(img_loader,
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=True,
                                                  num_workers=WORKERS,
                                                  pin_memory=True)

        angular_error = validate(test_loader, model, criterion, 100, compute_angular_error_fn, sum_writer)
        print('Angular Error on Test set:', angular_error)
        return

    assert not os.path.exists(checkpoint_fpath)

    for epoch in range(0, EPOCHS):
        if epoch == unfreeze_epoch:
            model_v.unfreeze_all()

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, compute_angular_error_fn, sum_writer)
        # evaluate on validation set
        angular_error, _ = validate(val_loader, model, criterion, epoch, compute_angular_error_fn, sum_writer)
        # remember best angular error in validation and save checkpoint
        is_best = angular_error < BEST_ERROR
        BEST_ERROR = min(angular_error, BEST_ERROR)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': BEST_ERROR,
        }, is_best, checkpoint_fpath)


def parse_cropsize(inp_str):
    if inp_str == '':
        return None
    return [int(x) for x in inp_str.split(',')]


def parse_dict(dict_str):
    """
    , => delimiter
    : => key value separator.

    """
    tokens = dict_str.split(',')
    output = {}
    for token in tokens:
        k, v = token.split(':')
        output[k] = v
    return output


if __name__ == '__main__':

    print(socket.gethostname(), datetime.now().strftime("%y-%m-%d-%H:%M:%S"))
    print('Python version', sys.version)
    print('CUDA_HOME', CUDA_HOME)
    print('CudaToolKit Version', torch.version.cuda)
    print('torch Version', torch.__version__)
    print('torchvision Version', torchvision.__version__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=ModelType.from_name)
    parser.add_argument('--backbone_type', type=BackboneType.from_name, default=BackboneType.Resnet18)
    parser.add_argument('--source_path', type=str, default='/tmp2/ashesh/gaze360_data/imgs/')
    parser.add_argument('--checkpoints_path', type=str, default='/home/ashesh/')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--evaluate_on', type=str, default='test.txt')
    parser.add_argument('--kfold', type=int, default=-1)
    parser.add_argument('--version', type=int, default=1)
    parser.add_argument('--atype', type=AggregationType.from_name, default=AggregationType.SPATIAL_MAX)
    parser.add_argument('--cropsize_list', type=parse_cropsize, default=None)
    parser.add_argument('--symmetric', type=int, default=0)
    parser.add_argument('--bidirectional', type=int, default=0)
    parser.add_argument('--enable_time', action='store_true')
    parser.add_argument('--dataloader_params', type=parse_dict, default=None)
    parser.add_argument('--sample_nbr_cnt', type=int, default=180)
    parser.add_argument('--agg_layer_idx', type=int)
    parser.add_argument('--magnify_ER_factor', type=float, default=1.5)
    args = parser.parse_args()
    assert args.model_type is not None

    cropsize_list = args.cropsize_list
    if cropsize_list is None:
        if args.enable_time:
            cropsize_list = [224, 200, 175, 150, 175, 200, 224]
        else:
            cropsize_list = [224, 200, 175, 150]

    params = {
        'evaluate': args.evaluate,
        'version': args.version,
        'atype': args.atype,
        'backbone_type': args.backbone_type,
        'cropsize_list': cropsize_list,
        'symmetric': args.symmetric,
        'bidirectional_lstm': args.bidirectional,
        'enable_time': args.enable_time,
        'dataloader_params': args.dataloader_params,
    }
    model_type = args.model_type

    if model_type == ModelType.Gaze360LazyAggregationModel:
        params['agg_layer_idx'] = args.agg_layer_idx

    train_file = 'train.txt'
    val_file = 'validation.txt'
    test_file = args.evaluate_on

    main(model_type, train_file, val_file, test_file, args.source_path, args.checkpoints_path, args.img_size,
         args.kfold, **params)
