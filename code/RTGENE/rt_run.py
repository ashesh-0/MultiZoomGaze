import argparse
import os
import socket
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.utils.cpp_extension import CUDA_HOME

from backbones.backbone_type import BackboneType
from core.aggregation_type import AggregationType
from core.early_stopping import EarlyStop
from core.enum import Enum
from core.model_type import ModelType
from core.prediction_utils import get_df_from_predictions, get_prediction
from core.train_utils import (AverageMeter, checkpoint_fname, compute_angular_error, compute_angular_error_arr,
                              save_checkpoint)
from model import GazeLSTM
from non_lstm_based_model import GazeMultiCropModel
from RTGENE.data_utils import DataType
from RTGENE.rt_data_loader import RTGeneImageLoaderLstmModel
from RTGENE.rt_data_loader_static import RTGeneImagerLoaderStaticModel
from static_model import GazeStaticModel


class EvaluateOn(Enum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2


workers = 4
epochs = 100
batch_size = 64
best_error = 100  # init with a large value
lr = 0.000325


def get_evaluation_data(img_loader, model, criterion, batch_size, workers):

    (prediction, actual) = get_prediction(model,
                                          img_loader,
                                          num_workers=workers,
                                          batch_size=batch_size,
                                          multi_output_model=False)

    error = compute_angular_error_arr(torch.Tensor(prediction), torch.Tensor(actual)).reshape(-1, 1)
    df = pd.DataFrame(np.concatenate([prediction, actual, error], axis=1),
                      columns=['pred_yaw', 'pred_pitch', 'yaw', 'pitch', 'error'])
    return df


def train(train_loader, model, criterion, optimizer, epoch, compute_angular_error_fn):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    angular = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (source_frame, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if isinstance(source_frame, list):
            N = source_frame[0].size(0)
            source_frame = [f.cuda(non_blocking=True) for f in source_frame]
            source_frame_var = [torch.autograd.Variable(f) for f in source_frame]
        else:
            N = source_frame.size(0)
            source_frame = source_frame.cuda(non_blocking=True)
            source_frame_var = torch.autograd.Variable(source_frame)

        target = target.cuda(non_blocking=True)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(source_frame_var)
        loss = criterion(output, target_var)

        angular_error = compute_angular_error_fn(output, target_var)

        angular.update(angular_error.item(), N)

        losses.update(loss.item(), N)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 100 == 0:
            # print('LR:', scheduler.get_lr())
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Angular {angular.val:.3f} ({angular.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      angular=angular,
                  ))


def validate(val_loader, model, criterion, epoch, compute_angular_error_fn, verbose=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    model.eval()
    end = time.time()
    angular = AverageMeter()
    for i, (source_frame, target) in enumerate(val_loader):

        if isinstance(source_frame, list):
            N = source_frame[0].size(0)
            source_frame = [f.cuda(non_blocking=True) for f in source_frame]
            source_frame_var = [torch.autograd.Variable(f) for f in source_frame]
        else:
            N = source_frame.size(0)
            source_frame = source_frame.cuda(non_blocking=True)
            source_frame_var = torch.autograd.Variable(source_frame)

        target = target.cuda(non_blocking=True)
        target_var = torch.autograd.Variable(target)
        with torch.no_grad():
            # compute output
            output = model(source_frame_var)

            loss = criterion(output, target_var)
            angular_error = compute_angular_error_fn(output, target_var)

            angular.update(angular_error.item(), N)

            losses.update(loss.item(), N)

            batch_time.update(time.time() - end)
            end = time.time()
            if verbose:
                print(f'Validation:{i}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Angular {angular.val:.4f} ({angular.avg:.4f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i,
                                                                      len(val_loader),
                                                                      batch_time=batch_time,
                                                                      loss=losses,
                                                                      angular=angular))

    print('Validation\t'
          'Time ({batch_time.avg:.3f})\t'
          'Angular ({angular.avg:.4f})\t'
          'Loss ({loss.avg:.4f})\t'.format(batch_time=batch_time, loss=losses, angular=angular))
    return (angular.avg, losses.avg)


def main(model_type, data_type, backbone_type, source_path, checkpoints_path, img_size, kfold_train, **params):
    global epochs
    global args, best_error
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

    assert model_type in [ModelType.StaticRTGENEModel, ModelType.RTGENEModel, ModelType.NonLstmRTGENEModel]
    assert kfold_train in [0, 1, 2]
    fold_dict = {
        0: [f"s{i:03d}_glasses" for i in [1, 2, 8, 10]],
        1: [f"s{i:03d}_glasses" for i in [3, 4, 7, 9]],
        2: [f"s{i:03d}_glasses" for i in [5, 6, 11, 12, 13]]
    }
    train_persons = fold_dict[(kfold_train + 1) % 3] + fold_dict[(kfold_train + 2) % 3]
    validation_persons = [f"s{i:03d}_glasses" for i in [0, 14, 15, 16]]

    # Evaluate on which set
    print('Evaluation set is set to ', EvaluateOn.name(params['evaluate_on']))
    if params['evaluate_on'] == EvaluateOn.TEST:
        test_persons = fold_dict[kfold_train]
    elif params['evaluate_on'] == EvaluateOn.VALIDATION:
        test_persons = validation_persons.copy()
    else:
        test_persons = train_persons.copy()

    compute_angular_error_fn = compute_angular_error
    criterion = torch.nn.MSELoss().cuda()
    train_img_loader_kwargs = {'data_type': data_type, 'person_list': train_persons}
    val_img_loader_kwargs = {'data_type': data_type, 'person_list': validation_persons}
    test_img_loader_kwargs = {'data_type': data_type, 'person_list': test_persons}

    if model_type == ModelType.StaticRTGENEModel:
        print('Using static model for RTGENE')
        ckp_tuples = [('TYPE', model_type)]
        if data_type != DataType.Inpainted:
            ckp_tuples += [('dtype', data_type)]

        if params['early_stop'] < epochs:
            ckp_tuples += [('estop', params['early_stop'])]

        if backbone_type != BackboneType.Resnet18:
            ckp_tuples += [('bkb', backbone_type)]

        ckp_tuples += [('v', f'master_{params["version"]}')]
        checkpoint_fpath = checkpoint_fname(
            True,
            kfold_train,
            ckp_tuples,
            dirname=checkpoints_path,
        )
        print(checkpoint_fpath)
        model_v = GazeStaticModel(output_dim=2, backbone_type=backbone_type)
        ImageLoaderClass = RTGeneImagerLoaderStaticModel
    elif model_type in [ModelType.RTGENEModel, ModelType.NonLstmRTGENEModel]:
        atype = None
        if model_type == ModelType.RTGENEModel:
            print('Using LSTM model for RTGENE')
        else:
            print('Using Non-LSTM model for RTGENE')
            assert params['symmetric'] == 0
            atype = AggregationType.SPATIAL_MAX

        cropsize_list = params['cropsize_list']
        if params['symmetric'] == 1:
            cropsize_list = cropsize_list + cropsize_list[::-1][1:]
            seq_len = len(cropsize_list)
            target_seq_index = seq_len // 2
        else:
            seq_len = len(cropsize_list)
            target_seq_index = seq_len - 1

        train_img_transforms = None
        val_img_transforms = None
        train_img_loader_kwargs['cropsize_list'] = cropsize_list
        val_img_loader_kwargs['cropsize_list'] = cropsize_list
        test_img_loader_kwargs['cropsize_list'] = cropsize_list

        ckp_tuples = [('TYPE', model_type)]
        if data_type != DataType.Inpainted:
            ckp_tuples += [('dtype', data_type)]
        if atype is not None:
            ckp_tuples += [('atype', atype)]
        # NOTE comment out below tuple addition in case checkpoint is not found for older models.
        ckp_tuples += [
            ('diff_crop', f'{cropsize_list[0]}-{cropsize_list[seq_len//2]}'),
            ('seq_len', seq_len),
        ]
        if target_seq_index != seq_len // 2:
            ckp_tuples += [('tar_idx', target_seq_index)]

        if params['early_stop'] < epochs:
            ckp_tuples += [('estop', params['early_stop'])]

        if backbone_type != BackboneType.Resnet18:
            ckp_tuples += [('bkb', backbone_type)]

        ckp_tuples += [('v', f'master_{params["version"]}')]

        checkpoint_fpath = checkpoint_fname(
            False,
            kfold_train,
            ckp_tuples,
            dirname=checkpoints_path,
        )
        print(checkpoint_fpath)
        if model_type == ModelType.RTGENEModel:
            model_v = GazeLSTM(output_dim=2,
                               seq_len=seq_len,
                               backbone_type=backbone_type,
                               target_seq_index=target_seq_index)
        elif model_type == ModelType.NonLstmRTGENEModel:
            model_v = GazeMultiCropModel(output_dim=2,
                                         atype=atype,
                                         backbone_type=backbone_type,
                                         cropsize_list=cropsize_list)

        ImageLoaderClass = RTGeneImageLoaderLstmModel

    model = torch.nn.DataParallel(model_v).cuda()
    model.cuda()

    cudnn.benchmark = True
    image_loader = ImageLoaderClass(source_path, None, train_img_transforms, **train_img_loader_kwargs)

    train_loader = torch.utils.data.DataLoader(image_loader,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=workers,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(ImageLoaderClass(source_path, None, val_img_transforms,
                                                              **val_img_loader_kwargs),
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=workers,
                                             pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.95))

    print('Overall Adam Optimizer')

    if params['evaluate']:
        print('Skipping Training')
        checkpoint_fpath = os.path.join(os.path.dirname(checkpoint_fpath),
                                        'model_best_' + os.path.basename(checkpoint_fpath))
        assert os.path.exists(checkpoint_fpath), checkpoint_fpath
        checkpoint = torch.load(checkpoint_fpath)
        model.load_state_dict(checkpoint['state_dict'])

        print(f"Loaded from {checkpoint_fpath}\n Epoch:{checkpoint['epoch']}")
        img_loader = ImageLoaderClass(source_path, None, val_img_transforms, **test_img_loader_kwargs)
        df = get_evaluation_data(img_loader, model, None, batch_size, workers)
        print('Angular Error on Test set')
        print(df['error'].describe())
        return

    assert not os.path.exists(checkpoint_fpath)

    earlystop = EarlyStop(patience=params['early_stop'])
    if params['early_stop'] < epochs:
        print('EarlyStop applied')

    for epoch in range(0, epochs):
        if epoch == unfreeze_epoch:
            model_v.unfreeze_all()

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, compute_angular_error_fn)
        angular_error, val_loss = validate(val_loader, model, criterion, epoch, compute_angular_error_fn)
        is_best = angular_error < best_error
        best_error = min(angular_error, best_error)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_error,
        }, is_best, checkpoint_fpath)

        if earlystop(val_loss):
            print('Early stopping')
            return


def parse_cropsize(inp_str):
    return [int(x) for x in inp_str.split(',')]


if __name__ == '__main__':
    print(socket.gethostname(), datetime.now().strftime("%y-%m-%d-%H:%M:%S"))
    print('Python version', sys.version)
    print('CUDA_HOME', CUDA_HOME)
    print('CudaToolKit Version', torch.version.cuda)
    print('torch Version', torch.__version__)
    print('torchvision Version', torchvision.__version__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=ModelType.from_name)
    parser.add_argument('--checkpoint_file', type=str, default='')
    parser.add_argument('--source_path', type=str, default='/tmp2/ashesh/gaze360_data/imgs/')
    parser.add_argument('--checkpoints_path', type=str, default='/home/ashesh/')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--evaluate_on', type=EvaluateOn.from_name, default=EvaluateOn.TEST)
    parser.add_argument('--kfold', type=int, default=-1)
    parser.add_argument('--early_stop', type=int, default=5)
    parser.add_argument('--version', type=int, default=1)
    parser.add_argument('--data_type', type=DataType.from_name, default=DataType.OriginalLarge)
    parser.add_argument('--backbone_type', type=BackboneType.from_name, default=BackboneType.Resnet18)
    parser.add_argument('--cropsize_list', type=parse_cropsize)
    parser.add_argument('--symmetric', type=int, default=1)
    args = parser.parse_args()
    params = {
        'evaluate': args.evaluate,
        'early_stop': args.early_stop,
        'version': args.version,
        'cropsize_list': args.cropsize_list,
        'symmetric': args.symmetric,
        'evaluate_on': args.evaluate_on,
    }

    assert args.backbone_type is not None
    assert args.data_type is not None
    assert args.model_type is not None

    main(
        args.model_type,
        args.data_type,
        args.backbone_type,
        args.source_path,
        args.checkpoints_path,
        args.img_size,
        args.kfold,
        **params,
    )
