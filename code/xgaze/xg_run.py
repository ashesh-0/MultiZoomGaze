"""
Example:
    python xgaze/xg_run.py --checkpoints_path=/home/ashesh/checkpoints/ --model_type=XgazeNonLstmMultiCropModel
    --atype=SPATIAL_MAX --cropsize_list=224,200,175,150
"""

import argparse
import os
import socket
import sys
from collections import OrderedDict
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
from core.train_utils import checkpoint_fname, compute_angular_error, compute_angular_error_xyz, save_checkpoint
from non_lstm_based_model import GazeMultiCropModel
from run_utils import evaluate, train, validate
from sinecosine_model.non_lstm_based_model import AggregationType
from sinecosine_model.static_sinecosine_model import GazeStaticSineAndCosineModel
from sinecosine_model.train_utils import compute_angular_error_sine_and_cosine
from static_model import GazeStaticModel
from xgaze.additive_model import AdditiveModel
from xgaze.evaluate import generate_test_csv
from xgaze.multicrop_model_separate_uncertainty import GazeMultiCropSepModel
from xgaze.static_model_separate_uncertainty import GazeStaticSepModel
from xgaze.xgaze_dataloader import get_trainset

lr = 1e-4
epochs = 15
workers = 4


def main(data_dir, checkpoints_path, model_type=ModelType.NonLstmMultiCropModel, **params):
    best_error = 10000
    cropsizes = params['cropsize_list']
    kfold_id = params.get('kfold_id', None)
    batch_size = params['batch_size']
    train_dset = get_trainset(data_dir,
                              model_type,
                              cropsizes=cropsizes,
                              is_shuffle=False,
                              is_validation=False,
                              kfold_id=kfold_id)
    val_dset = get_trainset(data_dir,
                            model_type,
                            cropsizes=cropsizes,
                            is_shuffle=False,
                            is_validation=True,
                            kfold_id=kfold_id)
    compute_angular_error_fn = compute_angular_error
    criterion = PinBallLoss().cuda()

    if model_type in [ModelType.XgazeStaticModel, ModelType.XgazeStaticCLModel]:
        backbone_type = params['backbone_type']
        checkpoint_fpath = checkpoint_fname(
            True,
            kfold_id,
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

        model_v = GazeStaticModel(backbone_type=backbone_type)
    elif model_type == ModelType.XgazeStaticSepModel:
        backbone_type = params['backbone_type']
        checkpoint_fpath = checkpoint_fname(
            True,
            kfold_id,
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

        model_v = GazeStaticSepModel(backbone_type=backbone_type)

    elif model_type == ModelType.XgazeStaticSCModel:
        backbone_type = params['backbone_type']
        checkpoint_fpath = checkpoint_fname(
            True,
            kfold_id,
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
        model_v = GazeStaticSineAndCosineModel(backbone_type=backbone_type)
        train_dset = get_trainset(data_dir,
                                  model_type,
                                  sc_target=True,
                                  cropsizes=cropsizes,
                                  is_shuffle=False,
                                  is_validation=False,
                                  kfold_id=kfold_id)
        val_dset = get_trainset(data_dir,
                                model_type,
                                sc_target=True,
                                cropsizes=cropsizes,
                                is_shuffle=False,
                                is_validation=True,
                                kfold_id=kfold_id)
        compute_angular_error_fn = compute_angular_error_sine_and_cosine

    elif model_type == ModelType.XgazeNonLstmMultiCropModel:
        print("Using Non lstm based MultiCrops with Gaze360 target and loss")
        backbone_type = params['backbone_type']
        # cropsize_list = None
        # target_seq_index = None
        seq_len = len(cropsizes)

        atype = params['atype']
        assert atype is not None

        checkpoint_fpath = checkpoint_fname(
            False,
            kfold_id,
            [
                ('TYPE', model_type),
                ('bkb', backbone_type),
                ('diff_crop', f'{cropsizes[0]}-{cropsizes[seq_len//2]}'),
                ('seq_len', seq_len),
                ('atype', atype),
                ('bsz', batch_size),
                ('lr', lr),
                ('v', f'master_{params["version"]}'),
            ],
            dirname=checkpoints_path,
        )
        print(checkpoint_fpath)
        model_v = GazeMultiCropModel(
            output_dim=3,
            backbone_type=backbone_type,
            cropsize_list=cropsizes,
            atype=atype,
        )
    elif model_type == ModelType.XgazeNonLstmMultiCropSepModel:
        print("Using Non lstm based MultiCrops with Gaze360 target and loss")
        backbone_type = params['backbone_type']
        # cropsize_list = None
        # target_seq_index = None
        seq_len = len(cropsizes)

        atype = params['atype']
        assert atype is not None

        checkpoint_fpath = checkpoint_fname(
            False,
            kfold_id,
            [
                ('TYPE', model_type),
                ('bkb', backbone_type),
                ('diff_crop', f'{cropsizes[0]}-{cropsizes[seq_len//2]}'),
                ('seq_len', seq_len),
                ('atype', atype),
                ('bsz', batch_size),
                ('lr', lr),
                ('v', f'master_{params["version"]}'),
            ],
            dirname=checkpoints_path,
        )
        print(checkpoint_fpath)
        model_v = GazeMultiCropSepModel(backbone_type=backbone_type, cropsize_list=cropsizes, atype=atype)
    elif model_type == ModelType.XgazeMultiCropAdditiveModel:
        # print("Using Non lstm based MultiCrops with Gaze360 target and loss")
        backbone_type = params['backbone_type']
        seq_len = len(cropsizes)
        # assert atype is None

        checkpoint_fpath = checkpoint_fname(
            False,
            kfold_id,
            [
                ('TYPE', model_type),
                ('bkb', backbone_type),
                ('diff_crop', f'{cropsizes[0]}-{cropsizes[seq_len//2]}'),
                ('seq_len', seq_len),
                ('bsz', batch_size),
                ('lr', lr),
                ('v', f'master_{params["version"]}'),
            ],
            dirname=checkpoints_path,
        )
        print(checkpoint_fpath)
        model_v = AdditiveModel(backbone_type=backbone_type)
    elif model_type == ModelType.XgazeStaticPairPretrainedModel:
        backbone_type = params['backbone_type']
        pretrained_fpath = params['p_fpath']
        assert len(pretrained_fpath) > 0 and os.path.exists(pretrained_fpath)
        checkpoint_fpath = checkpoint_fname(
            True,
            kfold_id,
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

        model_v = GazeStaticModel(backbone_type=backbone_type)
        last_layer = model_v.last_layer
        model_v.last_layer = nn.Linear(model_v.img_feature_dim, 2)
        checkpoint = torch.load(pretrained_fpath)

        def new_key(key):
            unwanted_pref = 'module.'
            assert unwanted_pref == key[:len(unwanted_pref)]
            return key[len(unwanted_pref):]

        state_dict = OrderedDict([(new_key(k), v) for k, v in checkpoint['state_dict'].items()])
        model_v.load_state_dict(state_dict)
        model_v.last_layer = last_layer
        print(f'Loaded weights from {pretrained_fpath}, Epoch:{checkpoint["epoch"]}')

        # train_from = 'layer4'
        # # train_from = 'fc1'
        # is_trainable = False
        # for name, parameter in model_v.named_parameters():
        #     is_trainable = train_from in name or is_trainable
        #     parameter.requires_grad = is_trainable
        #     if is_trainable:
        #         print('Trainable', name)

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
        if model_type == ModelType.XgazeStaticCLModel:
            if epoch == 1:
                train_dset.set_leftright(train_dset.RIGHT)
                val_dset.set_leftright(val_dset.RIGHT)
            elif epoch == 2:
                train_dset.set_leftright(train_dset.BOTH)
                val_dset.set_leftright(val_dset.BOTH)

        for param_group in optimizer.param_groups:
            print('Epoch:', epoch, 'LR:', param_group['lr'])
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, compute_angular_error_fn)
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
    parser.add_argument('--kfold', type=int, default=-1)
    parser.add_argument('--pretrained_fpath', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=32)
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
        'p_fpath': args.pretrained_fpath,
        'batch_size': args.batch_size,
    }
    if args.kfold != -1:
        params['kfold_id'] = args.kfold
    model_type = args.model_type
    assert (model_type in [ModelType.XgazeNonLstmMultiCropModel, ModelType.XgazeNonLstmMultiCropSepModel
                           ]) != (args.atype == -1), ('Aggregation only defined '
                                                      'for NonLstm multicrop models and is necessary')

    main(args.source_path, args.checkpoints_path, model_type=model_type, **params)
