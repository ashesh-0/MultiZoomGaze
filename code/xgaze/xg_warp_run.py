import argparse
import math
import os
import socket
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
from torch.optim.lr_scheduler import StepLR
from torch.utils.cpp_extension import CUDA_HOME

from backbones.backbone_type import BackboneType
from core.aggregation_type import AggregationType
from core.loss import PinBallLoss
from core.model_type import ModelType
from core.train_utils import AverageMeter, checkpoint_fname, compute_angular_error, save_checkpoint
from xgaze.evaluate import generate_test_csv
from xgaze.loss import ExpWarp2Loss, ExpWarp3Loss, ExpWarp4Loss, ExpWarp5Loss, ExpWarp6Loss, ExpWarpLoss
from xgaze.static_warping_model import GazeStaticWarpModel
from xgaze.xgaze_dataloader import get_trainset

batch_size = 32
lr = 1e-4
epochs = 15
workers = 4


def validate(val_loader,
             model,
             criterion,
             warp_criterion,
             epoch,
             compute_angular_error_fn,
             summary_writer=None,
             verbose=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    warp_losses = AverageMeter()
    prediction_error = AverageMeter()
    model.eval()
    end = time.time()
    angular = AverageMeter()
    for i, (source_frame, target) in enumerate(val_loader):

        N = source_frame.size(0)
        source_frame = source_frame.cuda(non_blocking=True)
        source_frame_var = torch.autograd.Variable(source_frame)

        target = target.cuda(non_blocking=True)
        target_var = torch.autograd.Variable(target)
        with torch.no_grad():
            # compute output
            output, ang_error = model(source_frame_var)
            ang_error = ang_error[:, 0]
            actual_output = output[:, 0]
            lwarp_output = output[:, 1]
            rwarp_output = output[:, 2]

            warp_loss = warp_criterion(actual_output, lwarp_output, rwarp_output, target_var)

            loss = criterion(actual_output, target_var, ang_error)
            angular_error = compute_angular_error_fn(actual_output, target_var)
            pred_error = ang_error[:, 0] * 180 / math.pi
            pred_error = torch.mean(pred_error, 0)

            angular.update(angular_error.item(), N)
            prediction_error.update(pred_error.item(), N)

            losses.update(loss.item(), N)
            warp_losses.update(warp_loss.item(), N)

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
          'Loss ({loss.avg:.4f})\t'
          'WLoss ({wloss.avg:.4f})\t'.format(batch_time=batch_time, loss=losses, wloss=warp_losses, angular=angular))
    if summary_writer is not None:
        summary_writer.add_scalar("predicted error", prediction_error.avg, epoch)
        summary_writer.add_scalar("angular-test", angular.avg, epoch)
        summary_writer.add_scalar("loss-test", losses.avg, epoch)
    return (angular.avg, losses.avg)


def train(
    train_loader,
    model,
    criterion,
    warp_criterion,
    optimizer,
    epoch,
    compute_angular_error_fn,
    warp_weight,
    summary_writer=None,
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    warp_losses = AverageMeter()
    prediction_error = AverageMeter()
    angular = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (source_frame, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        N = source_frame.size(0)
        source_frame = source_frame.cuda(non_blocking=True)
        source_frame_var = torch.autograd.Variable(source_frame)

        target = target.cuda(non_blocking=True)
        target_var = torch.autograd.Variable(target)

        # compute output
        output, ang_error = model(source_frame_var)
        ang_error = ang_error[:, 0]
        actual_output = output[:, 0]
        lwarp_output = output[:, 1]
        rwarp_output = output[:, 2]

        warp_loss = warp_criterion(actual_output, lwarp_output, rwarp_output, target_var)
        loss = criterion(actual_output, target_var, ang_error)
        angular_error = compute_angular_error_fn(actual_output, target_var)
        pred_error = ang_error[:, 0] * 180 / math.pi
        pred_error = torch.mean(pred_error, 0)

        angular.update(angular_error.item(), N)

        warp_losses.update(warp_loss.item(), N)
        losses.update(loss.item(), N)

        prediction_error.update(pred_error.item(), N)
        if summary_writer is not None:
            summary_writer.add_scalar("loss", losses.val, epoch)
            summary_writer.add_scalar("angular", angular.val, epoch)

        loss += warp_weight * warp_loss
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
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'WLoss {wloss.val:.4f} ({wloss.avg:.4f})\t'
                  'Prediction Error {prediction_error.val:.4f} ({prediction_error.avg:.4f})\t'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      wloss=warp_losses,
                      angular=angular,
                      prediction_error=prediction_error))


def main(data_dir, checkpoints_path, model_type=ModelType.NonLstmMultiCropModel, **params):
    best_error = 10000
    cropsizes = params['cropsize_list']
    kfold_id = params.get('kfold_id', None)
    train_dset = get_trainset(data_dir,
                              model_type,
                              cropsizes=cropsizes,
                              yaw_warping_angle=params['yaw_warping_angle'],
                              pitch_warping_angle=params['pitch_warping_angle'],
                              is_shuffle=False,
                              is_validation=False,
                              kfold_id=kfold_id)
    val_dset = get_trainset(data_dir,
                            model_type,
                            cropsizes=cropsizes,
                            yaw_warping_angle=params['yaw_warping_angle'],
                            pitch_warping_angle=params['pitch_warping_angle'],
                            is_shuffle=False,
                            is_validation=True,
                            kfold_id=kfold_id)
    compute_angular_error_fn = compute_angular_error
    criterion = PinBallLoss().cuda()
    if model_type == ModelType.XgazeStaticWarpModel:
        warp_criterion = ExpWarpLoss(params['yaw_warping_angle'], params['pitch_warping_angle']).cuda()
    elif model_type == ModelType.XgazeStaticWarp3Model:
        warp_criterion = ExpWarp3Loss(params['yaw_warping_angle'], params['pitch_warping_angle']).cuda()
    elif model_type == ModelType.XgazeStaticWarp4Model:
        warp_criterion = ExpWarp4Loss(params['yaw_warping_angle'], params['pitch_warping_angle']).cuda()
    elif model_type == ModelType.XgazeStaticWarp2Model:
        warp_criterion = ExpWarp2Loss(params['yaw_warping_angle'], params['pitch_warping_angle']).cuda()
    elif model_type == ModelType.XgazeStaticWarp5Model:
        yaw_range = [(-2 * math.pi / 3, -math.pi / 3), (math.pi / 3, 2 * math.pi / 3)]
        warp_criterion = ExpWarp5Loss(params['yaw_warping_angle'], params['pitch_warping_angle'], yaw_range,
                                      None).cuda()
    elif model_type == ModelType.XgazeStaticWarp6Model:
        mean = 5 / 180 * math.pi
        std = 1 / 180 * math.pi
        warp_criterion = ExpWarp6Loss(params['yaw_warping_angle'], params['pitch_warping_angle'], mean, std).cuda()
    assert model_type in [
        ModelType.XgazeStaticWarpModel, ModelType.XgazeStaticWarp3Model, ModelType.XgazeStaticWarp4Model,
        ModelType.XgazeStaticWarp2Model, ModelType.XgazeStaticWarp5Model, ModelType.XgazeStaticWarp6Model
    ]
    backbone_type = params['backbone_type']
    warp_weight = params['warp_weight']
    ckp_lists = [
        ('TYPE', model_type),
        ('bkb', backbone_type),
    ]
    if warp_weight != 1:
        ckp_lists += [('ww', warp_weight)]

    ckp_lists += [
        ('Ywarp', params['yaw_warping_angle']),
        ('Pwarp', params['pitch_warping_angle']),
        ('bsz', batch_size),
        ('lr', lr),
        ('v', f'master_{params["version"]}'),
    ]
    checkpoint_fpath = checkpoint_fname(
        True,
        kfold_id,
        ckp_lists,
        dirname=checkpoints_path,
    )
    print(checkpoint_fpath)

    model_v = GazeStaticWarpModel(backbone_type=backbone_type)
    model = torch.nn.DataParallel(model_v).cuda()
    model.cuda()

    # cudnn.benchmark = True
    # import pdb
    # pdb.set_trace()
    train_loader = torch.utils.data.DataLoader(train_dset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=workers,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=workers,
                                             pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    print('Overall Adam Optimizer')

    if params['evaluate']:
        print('Skipping Training')
        checkpoint_fpath = os.path.join(os.path.dirname(checkpoint_fpath),
                                        'model_best_' + os.path.basename(checkpoint_fpath))
        assert os.path.exists(checkpoint_fpath), checkpoint_fpath
        checkpoint = torch.load(checkpoint_fpath)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded from {checkpoint_fpath}\n Epoch:{checkpoint['epoch']}")
        generate_test_csv(model,
                          data_dir,
                          model_type,
                          cropsizes=cropsizes,
                          kfold_id=kfold_id,
                          batch_size=batch_size,
                          workers=workers)
        return

    assert not os.path.exists(checkpoint_fpath)

    for epoch in range(0, epochs):

        # train for one epoch
        train(train_loader, model, criterion, warp_criterion, optimizer, epoch, compute_angular_error_fn, warp_weight)
        # evaluate on validation set
        angular_error, loss = validate(val_loader, model, criterion, warp_criterion, epoch, compute_angular_error_fn)
        # scheduler.step(angular_error)
        # remember best angular error in validation and save checkpoint
        is_best = angular_error < best_error
        best_error = min(angular_error, best_error)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_error,
        }, is_best, checkpoint_fpath)
        scheduler.step()


def parse_cropsize(inp_str):
    if inp_str == '':
        return None
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
    parser.add_argument('--backbone_type', type=BackboneType.from_name, default=BackboneType.Resnet18)
    parser.add_argument('--source_path', type=str, default='/tmp2/ashesh/xgaze_224/')
    parser.add_argument('--checkpoints_path', type=str, default='/home/ashesh/')
    # parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--evaluate', action='store_true')
    # parser.add_argument('--evaluate_on', type=str, default='test.txt')
    # parser.add_argument('--kfold', type=int, default=-1)
    parser.add_argument('--version', type=int, default=1)
    parser.add_argument('--atype', type=AggregationType.from_name, default=-1)
    parser.add_argument('--cropsize_list', type=parse_cropsize)
    parser.add_argument('--symmetric', type=int, default=1)
    parser.add_argument('--bidirectional', type=int, default=0)
    parser.add_argument('--kfold', type=int, default=-1)
    parser.add_argument('--yaw_warping_angle', type=float, default=0.0)
    parser.add_argument('--pitch_warping_angle', type=float, default=0.0)
    parser.add_argument('--warp_weight', type=float, default=1.0)
    # parser.add_argument('--enable_time', action='store_true')
    args = parser.parse_args()
    assert args.model_type is not None

    params = {
        'evaluate': args.evaluate,
        'version': args.version,
        'atype': args.atype,
        'backbone_type': args.backbone_type,
        'cropsize_list': args.cropsize_list,
        'symmetric': args.symmetric,
        'bidirectional_lstm': args.bidirectional,
        'yaw_warping_angle': args.yaw_warping_angle,
        'pitch_warping_angle': args.pitch_warping_angle,
        'warp_weight': args.warp_weight,
    }
    if args.kfold != -1:
        params['kfold_id'] = args.kfold
    model_type = args.model_type
    assert (model_type in [ModelType.XgazeNonLstmMultiCropModel, ModelType.XgazeNonLstmMultiCropSepModel
                           ]) != (args.atype == -1), ('Aggregation only defined '
                                                      'for NonLstm multicrop models and is necessary')

    main(args.source_path, args.checkpoints_path, model_type=model_type, **params)
