import math
import time

import numpy as np
import pandas as pd
import torch

from core.analysis_utils import print_gaze360_metrics
from core.gaze_utils import compute_angular_error_arr
from core.prediction_utils import get_df_from_predictions, get_prediction, get_prediction_with_uncertainty
from core.train_utils import AverageMeter, compute_angular_error
from sinecosine_model.train_utils import (compute_angular_error_sine_and_cosine, get_cosinebased_yaw_pitch,
                                          get_sinebased_yaw_pitch)


def train(train_loader, model, criterion, optimizer, epoch, compute_angular_error_fn, summary_writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    prediction_error = AverageMeter()
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
        output, ang_error = model(source_frame_var)

        loss = criterion(output, target_var, ang_error)

        angular_error = compute_angular_error_fn(output, target_var)
        pred_error = ang_error[:, 0] * 180 / math.pi
        pred_error = torch.mean(pred_error, 0)

        angular.update(angular_error.item(), N)

        losses.update(loss.item(), N)

        prediction_error.update(pred_error.item(), N)
        if summary_writer is not None:
            summary_writer.add_scalar("loss", losses.val, epoch)
            summary_writer.add_scalar("angular", angular.val, epoch)
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
                  'Prediction Error {prediction_error.val:.4f} ({prediction_error.avg:.4f})\t'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      angular=angular,
                      prediction_error=prediction_error))


def get_evaluation_data(img_loader, model, criterion, compute_angular_error_fn, batch_size, workers):
    if compute_angular_error_fn not in [compute_angular_error, compute_angular_error_sine_and_cosine]:
        print('Evaluation not supported for this angular error method', compute_angular_error_fn)
        return None

    (prediction, actual, uncertainty) = get_prediction_with_uncertainty(model,
                                                                        img_loader,
                                                                        num_workers=workers,
                                                                        batch_size=batch_size)

    multiple_inputs = isinstance(img_loader.imgs[0][0], list)
    if compute_angular_error_fn == compute_angular_error:
        df = get_df_from_predictions(prediction, actual, img_loader, lstm_model=multiple_inputs)
        df['uncertainty'] = uncertainty[:, 0]
        return df

    elif compute_angular_error_fn == compute_angular_error_sine_and_cosine:
        pred_cos_yaw_pitch = get_cosinebased_yaw_pitch(torch.Tensor(prediction)).numpy()
        pred_sin_yaw_pitch = get_sinebased_yaw_pitch(torch.Tensor(prediction)).numpy()
        actual_sin_yaw_pitch = get_sinebased_yaw_pitch(torch.Tensor(actual)).numpy()
        actual_cos_yaw_pitch = get_cosinebased_yaw_pitch(torch.Tensor(actual)).numpy()
        assert np.abs(actual_sin_yaw_pitch - actual_cos_yaw_pitch).max() < 1e-03
        actual_yaw_pitch = actual_cos_yaw_pitch
        cos_df = get_df_from_predictions(pred_cos_yaw_pitch, actual_yaw_pitch, img_loader, lstm_model=multiple_inputs)
        sin_df = get_df_from_predictions(pred_sin_yaw_pitch, actual_yaw_pitch, img_loader, lstm_model=multiple_inputs)
        cos_df.rename({
            'pred_yaw': 'pred_yaw_cos',
            'pred_pitch': 'pred_pitch_cos',
            'angular_err': 'angular_err_cos'
        },
                      axis=1,
                      inplace=True)
        sin_df.rename({
            'pred_yaw': 'pred_yaw_sin',
            'pred_pitch': 'pred_pitch_sin',
            'angular_err': 'angular_err_sin'
        },
                      axis=1,
                      inplace=True)
        df = pd.merge(sin_df,
                      cos_df[['pred_yaw_cos', 'pred_pitch_cos', 'angular_err_cos', 'idx']],
                      how='outer',
                      on='idx')
        df['uncertainty'] = uncertainty[:, 0]
        assert not df.isna().any().any()
        df['sin_weight'] = ((df['pred_yaw_sin'] + df['pred_yaw_cos']) / 2).apply(math.cos).abs()
        df['pred_yaw_weighted'] = df['pred_yaw_sin'] * df['sin_weight'] + (1 - df['sin_weight']) * df['pred_yaw_cos']
        df['pred_yaw_avg'] = (df['pred_yaw_sin'] + df['pred_yaw_cos']) / 2
        df['angular_err_avg'] = compute_angular_error_arr(df[['pred_yaw_avg', 'pred_pitch_cos']].values,
                                                          df[['g_yaw', 'g_pitch']].values)
        df['angular_err_weighted'] = compute_angular_error_arr(df[['pred_yaw_weighted', 'pred_pitch_cos']].values,
                                                               df[['g_yaw', 'g_pitch']].values)

        return df


def evaluate(img_loader, model, criterion, compute_angular_error_fn, batch_size, workers):
    df = get_evaluation_data(img_loader, model, criterion, compute_angular_error_fn, batch_size, workers)
    if df is None:
        return False
    if 'angular_err_avg' in df.columns:
        print_gaze360_metrics(df, 'angular_err_avg')
        print_gaze360_metrics(df, 'angular_err_weighted')
    else:
        assert 'angular_err' in df.columns
        print_gaze360_metrics(df, 'angular_err')
    return True


def validate(val_loader, model, criterion, epoch, compute_angular_error_fn, summary_writer=None, verbose=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    prediction_error = AverageMeter()
    model.eval()
    end = time.time()
    angular = AverageMeter()
    for i, (source_frame, target) in enumerate(val_loader):

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
            output, ang_error = model(source_frame_var)

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
