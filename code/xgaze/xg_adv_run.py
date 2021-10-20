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
# from tensorboardX import SummaryWriter
from torch.utils.cpp_extension import CUDA_HOME

from backbones.backbone_type import BackboneType
from core.aggregation_type import AggregationType
from core.loss import PinBallLoss
from core.model_type import ModelType
from core.train_utils import (AverageMeter, checkpoint_fname, compute_angular_error, compute_angular_error_xyz,
                              save_checkpoint)
from non_lstm_based_model import GazeMultiCropModel
# from run_utils import evaluate, train, validate
from sinecosine_model.non_lstm_based_model import AggregationType
from sinecosine_model.static_sinecosine_model import GazeStaticSineAndCosineModel
from sinecosine_model.train_utils import compute_angular_error_sine_and_cosine
from xgaze.evaluate import generate_test_csv
from xgaze.static_model_adverserial import AdverserialBranch, GazeStaticAdverserialModel
from xgaze.xgaze_dataloader import get_trainset

batch_size = 50
lr = 1e-4
epochs = 15
workers = 4


def validate(val_loader, model, criterion, epoch, compute_angular_error_fn, summary_writer=None, verbose=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    prediction_error = AverageMeter()
    model.eval()
    end = time.time()
    angular = AverageMeter()
    for i, (source_frame, target, _) in enumerate(val_loader):

        # source_frame = source_frame.cuda(non_blocking=True)
        # source_frame_var = torch.autograd.Variable(source_frame)
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
            output, ang_error, _ = model(source_frame_var)

            loss = criterion(output, target_var, ang_error)
            angular_error = compute_angular_error_fn(output, target_var)
            pred_error = ang_error[:, 0] * 180 / math.pi
            pred_error = torch.mean(pred_error, 0)

            angular.update(angular_error.item(), N)
            prediction_error.update(pred_error.item(), N)

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
    if summary_writer is not None:
        summary_writer.add_scalar("predicted error", prediction_error.avg, epoch)
        summary_writer.add_scalar("angular-test", angular.avg, epoch)
        summary_writer.add_scalar("loss-test", losses.avg, epoch)
    return (angular.avg, losses.avg)


def train(train_loader,
          model,
          adv_model,
          criterion,
          criterion_adv,
          optimizer,
          optimizer_adv,
          epoch,
          compute_angular_error_fn,
          alpha=-0.0001,
          summary_writer=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    prediction_error = AverageMeter()
    angular = AverageMeter()
    adv_losses = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (source_frame, target, target_id) in enumerate(train_loader):
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

        target_id = target_id.cuda(non_blocking=True)
        target_id_var = torch.autograd.Variable(target_id)

        # compute output
        output, ang_error, embedding = model(source_frame_var)
        adv_output = adv_model(embedding)

        loss = criterion(output, target_var, ang_error)
        loss_adv = criterion_adv(adv_output, target_id_var)

        angular_error = compute_angular_error_fn(output, target_var)
        pred_error = ang_error[:, 0] * 180 / math.pi
        pred_error = torch.mean(pred_error, 0)

        angular.update(angular_error.item(), N)

        losses.update(loss.item(), N)
        adv_losses.update(loss_adv.item(), N)

        prediction_error.update(pred_error.item(), N)
        if summary_writer is not None:
            summary_writer.add_scalar("loss", losses.val, epoch)
            summary_writer.add_scalar("angular", angular.val, epoch)

        # compute gradient and do SGD step
        loss = loss - alpha * loss_adv
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # Adverserial training
        optimizer_adv.zero_grad()
        loss_adv.backward()
        optimizer_adv.step()

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
                  'AdvLoss{advloss.val:.4f} ({advloss.avg:.4f})\t'
                  'Prediction Error {prediction_error.val:.4f} ({prediction_error.avg:.4f})\t'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      advloss=adv_losses,
                      angular=angular,
                      prediction_error=prediction_error))


def main(data_dir, checkpoints_path, model_type=ModelType.NonLstmMultiCropModel, **params):
    kfold_train = -1
    best_error = 10000
    cropsizes = params['cropsize_list']
    train_dset = get_trainset(data_dir, model_type, cropsizes=cropsizes, is_shuffle=False, is_validation=False)
    val_dset = get_trainset(data_dir, model_type, cropsizes=cropsizes, is_shuffle=False, is_validation=True)
    compute_angular_error_fn = compute_angular_error
    criterion = PinBallLoss().cuda()
    criterion_adv = nn.CrossEntropyLoss()
    if model_type == ModelType.XgazeStaticAdvModel:
        backbone_type = params['backbone_type']
        checkpoint_fpath = checkpoint_fname(
            True,
            kfold_train,
            [
                ('TYPE', model_type),
                ('bkb', backbone_type),
                ('bsz', batch_size),
                ('lr', lr),
                ('v', f'master_{params["version"]}'),
            ],
            dirname=checkpoints_path,
        )
        print(checkpoint_fpath)

        model_v = GazeStaticAdverserialModel(backbone_type=backbone_type)
        adv_model_v = AdverserialBranch()

    model = torch.nn.DataParallel(model_v).cuda()
    model.cuda()

    adv_model = torch.nn.DataParallel(adv_model_v).cuda()
    adv_model.cuda()

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
    optimizer_adv = torch.optim.Adam(adv_model.parameters(), lr)

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
        generate_test_csv(model, data_dir, model_type, cropsizes=cropsizes, batch_size=batch_size, workers=workers)
        return

    assert not os.path.exists(checkpoint_fpath)

    for epoch in range(0, epochs):
        for param_group in optimizer.param_groups:
            print('Epoch:', epoch, 'LR:', param_group['lr'])
        # train for one epoch
        train(train_loader, model, adv_model, criterion, criterion_adv, optimizer, optimizer_adv, epoch,
              compute_angular_error_fn)
        # evaluate on validation set
        angular_error, loss = validate(val_loader, model, criterion, epoch, compute_angular_error_fn)
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
    }
    model_type = args.model_type
    assert (model_type in [
        ModelType.XgazeNonLstmMultiCropModel,
    ]) != (args.atype == -1), ('Aggregation only defined '
                               'for NonLstm multicrop models and is necessary')

    main(args.source_path, args.checkpoints_path, model_type=model_type, **params)
