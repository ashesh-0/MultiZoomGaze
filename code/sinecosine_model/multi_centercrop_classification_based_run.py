import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from core.pickle_utils import savePickle
from core.train_utils import AverageMeter, checkpoint_params
from sinecosine_model.data_loader_static_heatmap import ImageLoaderStaticHeatmap, get_personid_dict, get_target_dict
from sinecosine_model.static_heatmap_classifier import StaticHeatmapClassifier

lr = 1e-4
epochs = 10
batch_size = 256
workers = 6

val_file = "/home/ashesh/code/Gaze360/code/validation.txt"
val_error_file = "/home/ashesh/notebook/sincos_model/static_model_centercrop_error.csv"

test_file = "/home/ashesh/code/Gaze360/code/test.txt"
test_error_file = "/home/ashesh/notebook/sincos_model/static_model_centercrop_error_testdata.csv"
personid_pkl = '/home/ashesh/notebook/FinalPersonId.pkl'

source_path = "/tmp2/ashesh/gaze360_data/imgs"
checkpoint_test = '/home/ashesh/gaze360_static_TYPE:9_fc1:None_fc2:128_freeze:62_unfreeze:1.pth.tar'
centercrop_cols = ['Error_crop_100', 'Error_crop_224']

test_cropprediction_file = "/home/ashesh/notebook/sincos_model/cropprediction_"
f"{'_'.join([c.split('_')[-1] for c in centercrop_cols])}.pkl"

CLASS1_THRESHOLDS = [0.1, 0.4, 0.43, 0.46, 0.5]


def get_model():
    params = checkpoint_params(checkpoint_test)
    model_v = StaticHeatmapClassifier(
        len(centercrop_cols), checkpoint_test, model_kwargs={
            'fc1': None,
            'fc2': int(params['fc2'])
        })
    model = torch.nn.DataParallel(model_v).cuda()
    _ = model.cuda()
    return model


def calc_accuracy(output, Y, class1_threshold):
    output = nn.Softmax(dim=-1)(output)
    prediction = (output[:, 1] > class1_threshold).type(torch.int8)
    # import pdb
    # pdb.set_trace()
    # _, max_indices = torch.max(output, 1)
    acc = (prediction == Y).sum().item() / prediction.size()[0]
    return acc


def get_output(source_frame, target, model, criterion, class1_threshold_list=[0.3]):
    source_frame = source_frame.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)
    source_frame_var = torch.autograd.Variable(source_frame)
    target_var = torch.autograd.Variable(target)
    output = model(source_frame_var)
    loss = criterion(output, target_var)
    acc_list = [calc_accuracy(output, target_var, c) for c in class1_threshold_list]
    return {'loss': loss, 'acc': acc_list, 'output': output}


def evaluate(img_loader, model, criterion, class1_threshold_list=CLASS1_THRESHOLDS):
    tot_loss = AverageMeter()
    tot_acc_list = [AverageMeter() for _ in class1_threshold_list]
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for i, (source_frame, target) in enumerate(img_loader):
            data = get_output(source_frame, target, model, criterion, class1_threshold_list=class1_threshold_list)
            loss = data['loss']
            acc_list = data['acc']
            tot_loss.update(loss.item(), source_frame.size(0))
            for i in range(len(class1_threshold_list)):
                tot_acc_list[i].update(acc_list[i], source_frame.size(0))
            all_outputs.append(data['output'])

    return {'output': torch.cat(all_outputs, 0), 'loss': tot_loss.avg, 'acc': [acc.avg for acc in tot_acc_list]}


def train(train_loader, model, criterion, optimizer):
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    for i, (source_frame, target) in enumerate(train_loader):
        data = get_output(source_frame, target, model, criterion, class1_threshold_list=[0.3])
        loss = data['loss']
        acc = data['acc'][0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.update(loss.item(), source_frame.size(0))
        train_acc.update(acc, source_frame.size(0))
    return train_loss, train_acc


def compute_angular_error_from_prediction(cropprediction, img_loader, error_path):
    # test_df = pd.Series(np.argmax(test_result['output'].cpu().numpy(), axis=1)).to_frame('cropscale')
    assert np.max(np.abs(np.sum(cropprediction, axis=1) - 1)) < 1e-5
    err_list = []
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
        df = pd.Series((cropprediction[:, 1] > thresh).astype(int)).to_frame('cropscale')
        df['fpath'] = [i[0] for i in img_loader.imgs]
        df.set_index('fpath', inplace=True)
        error_df = pd.read_csv(error_path, index_col=0)
        error_df.set_index('fpath', inplace=True)
        error_df = error_df.join(df, how='outer')
        assert error_df.isna().any().sum() == 0
        # assert set(error_df.cropscale.unique()) == set([0, 1]), f'{error_df.cropscale.unique()}'
        print('Prediction contains following classes:\n', error_df.cropscale.value_counts(normalize=True))

        error_col0 = error_df[error_df.cropscale == 0][centercrop_cols[0]].sum()
        error_col1 = error_df[error_df.cropscale == 1][centercrop_cols[1]].sum()
        err = (error_col0 + error_col1) / error_df.shape[0]
        err_list.append(round(err, 2))
    return err_list
    # print(f'Test: Loss:{round(result["loss"],2)} Acc:{round(val_result["acc"],2)} AngularError:{err}')


def run():
    target_dict = get_target_dict(val_error_file, centercrop_cols)
    imbalance_dict = pd.Series([v for _, v in target_dict.items()]).value_counts(normalize=True)
    class_weights = torch.FloatTensor([imbalance_dict[1], imbalance_dict[0]]).cuda()
    # {0: imbalance_dict[1], 1: imbalance_dict[0]}
    print('ClassWeights', class_weights)
    # import pdb
    # pdb.set_trace()

    personid_dict = get_personid_dict(val_error_file, personid_pkl)

    test_target_dict = get_target_dict(test_error_file, centercrop_cols)
    test_personid_dict = get_personid_dict(test_error_file, personid_pkl)

    image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    personids = list(set([v for _, v in personid_dict.items()]))
    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        image_normalize,
    ])
    test_img_loader = ImageLoaderStaticHeatmap(test_target_dict, test_personid_dict, None, source_path, test_file,
                                               img_transforms)
    test_loader = torch.utils.data.DataLoader(
        test_img_loader, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    test_result_dict = {}
    for val_personid in personids:
        train_personids = sorted(list(set(personids) - set([val_personid])))
        print('')
        print('')
        print('Validation PersonId', val_personid)
        print('Train PersonId', train_personids)
        t_img_loader = ImageLoaderStaticHeatmap(target_dict, personid_dict, train_personids, source_path, val_file,
                                                img_transforms)
        v_img_loader = ImageLoaderStaticHeatmap(target_dict, personid_dict, [val_personid], source_path, val_file,
                                                img_transforms)

        model = get_model()
        optimizer = torch.optim.Adam(model.parameters(), lr)
        train_loader = torch.utils.data.DataLoader(
            t_img_loader, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            v_img_loader, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

        for epoch in range(epochs):
            train_loss, train_acc = train(train_loader, model, criterion, optimizer)
            val_result = evaluate(val_loader, model, criterion)
            acc = [round(x, 2) for x in val_result['acc']]
            print(f'[Epoch {epoch}] Train: Loss:{round(train_loss.avg,2)} Acc:{round(train_acc.avg,2)} '
                  f'Val: Loss:{round(val_result["loss"],2)} Acc:{acc}')

        # Test.
        test_result = evaluate(test_loader, model, criterion)
        acc = [round(x, 2) for x in test_result['acc']]
        err = compute_angular_error_from_prediction(
            nn.Softmax(dim=-1)(test_result['output']).cpu().numpy(), test_img_loader, test_error_file)
        print(f'Test: Loss:{round(test_result["loss"],2)} Acc:{acc} AngularError:{err}')
        # break
        # import pdb
        # pdb.set_trace()
    savePickle(test_cropprediction_file, test_result_dict)


if __name__ == '__main__':
    run()
