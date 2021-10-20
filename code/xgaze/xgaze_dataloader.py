import json
import os

from torchvision import transforms

from core.model_type import ModelType
from xgaze.train_val_split import get_train_val_split
from xgaze.xgaze_multicrop_dataloader import GazeDataset, GazeMulticropDataset
from xgaze.xgaze_static_dataloader import GazeDatasetCL, GazeDatasetWithId
from xgaze.xgaze_static_paired_dataloader import PairedDataLoader
from xgaze.xgaze_static_warping_dataloader import GazeDatasetWithWarping

trans_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_trainset(
    data_dir,
    model_type,
    sc_target=False,
    cropsizes=None,
    is_shuffle=True,
    is_validation=False,
    yaw_warping_angle=None,
    pitch_warping_angle=None,
    kfold_id=None,
    **kwargs,
):

    sub_folder_use = 'train'
    assert isinstance(is_validation, bool)
    is_validation = int(is_validation)
    assert sc_target is False or model_type == ModelType.XgazeStaticSCModel
    if model_type in [
            ModelType.XgazeStaticModel, ModelType.XgazeStaticSepModel, ModelType.XgazeStaticSCModel,
            ModelType.XgazeStaticPairPretrainedModel
    ]:
        train_set = GazeDataset(dataset_path=data_dir,
                                keys_to_use=get_train_val_split(kfold_id=kfold_id)[is_validation],
                                sub_folder=sub_folder_use,
                                sc_target=sc_target,
                                transform=trans,
                                is_shuffle=is_shuffle,
                                is_load_label=True)
    elif model_type in [
            ModelType.XgazeNonLstmMultiCropModel, ModelType.XgazeNonLstmMultiCropSepModel,
            ModelType.XgazeMultiCropAdditiveModel
    ]:
        train_set = GazeMulticropDataset(dataset_path=data_dir,
                                         keys_to_use=get_train_val_split(kfold_id=kfold_id)[is_validation],
                                         cropsizes=cropsizes,
                                         sub_folder=sub_folder_use,
                                         transform=trans,
                                         is_shuffle=is_shuffle,
                                         is_load_label=True)
    elif model_type == ModelType.XgazeStaticAdvModel:
        train_set = GazeDatasetWithId(dataset_path=data_dir,
                                      keys_to_use=get_train_val_split(kfold_id=kfold_id)[is_validation],
                                      sub_folder=sub_folder_use,
                                      sc_target=sc_target,
                                      transform=trans,
                                      is_shuffle=is_shuffle,
                                      is_load_label=True)
    elif model_type == ModelType.XgazeStaticCLModel:
        train_set = GazeDatasetCL(dataset_path=data_dir,
                                  keys_to_use=get_train_val_split(kfold_id=kfold_id)[is_validation],
                                  sub_folder=sub_folder_use,
                                  sc_target=sc_target,
                                  transform=trans,
                                  is_shuffle=is_shuffle,
                                  is_load_label=True)
    elif model_type in [
            ModelType.XgazeStaticWarpModel, ModelType.XgazeStaticWarp3Model, ModelType.XgazeStaticWarp4Model,
            ModelType.XgazeStaticWarp2Model, ModelType.XgazeStaticWarp5Model, ModelType.XgazeStaticWarp6Model
    ]:
        train_set = GazeDatasetWithWarping(dataset_path=data_dir,
                                           yaw_warping_angle=yaw_warping_angle,
                                           pitch_warping_angle=pitch_warping_angle,
                                           keys_to_use=get_train_val_split(kfold_id=kfold_id)[is_validation],
                                           sub_folder=sub_folder_use,
                                           sc_target=sc_target,
                                           transform=trans,
                                           is_shuffle=is_shuffle,
                                           is_load_label=True)
    elif model_type in [
            ModelType.XgazeStaticPairModel, ModelType.XgazeStaticPair2Model, ModelType.XgazeStaticPinballPairModel
    ]:
        bin_size = kwargs['bin_size']
        ignore_nbr_cnt = kwargs['ignore_nbr_cnt']
        sample_nbr_cnt = kwargs['sample_nbr_cnt']
        ignore_same_bucket = kwargs['ignore_same_bucket']
        train_set = PairedDataLoader(dataset_path=data_dir,
                                     keys_to_use=get_train_val_split(kfold_id=kfold_id)[is_validation],
                                     sub_folder=sub_folder_use,
                                     transform=trans,
                                     yaw_bin_size=bin_size,
                                     pitch_bin_size=bin_size,
                                     ignore_nbr_cnt=ignore_nbr_cnt,
                                     sample_nbr_cnt=sample_nbr_cnt,
                                     ignore_same_bucket=ignore_same_bucket,
                                     is_shuffle=is_shuffle,
                                     is_load_label=True)
    else:
        raise Exception('ModelType:{model_type} not supported')
    return train_set


def get_testset(data_dir,
                model_type,
                sc_target=False,
                yaw_warping_angle=None,
                pitch_warping_angle=None,
                cropsizes=None,
                **kwargs):
    # load dataset
    refer_list_file = os.path.join(data_dir, 'train_test_split.json')
    print('load the train file list from: ', refer_list_file)

    with open(refer_list_file, 'r') as f:
        datastore = json.load(f)

    # there are three subsets for ETH-XGaze dataset: train, test and test_person_specific
    # train set: the training set includes 80 participants data
    # test set: the test set for cross-dataset and within-dataset evaluations
    # test_person_specific: evaluation subset for the person specific setting
    sub_folder_use = 'test'
    if model_type in [
            ModelType.XgazeStaticModel, ModelType.XgazeStaticSepModel, ModelType.XgazeStaticPairPretrainedModel
    ]:
        test_set = GazeDataset(dataset_path=data_dir,
                               keys_to_use=datastore[sub_folder_use],
                               sub_folder=sub_folder_use,
                               sc_target=sc_target,
                               transform=trans,
                               is_shuffle=False,
                               is_load_label=False)

    elif model_type in [ModelType.XgazeNonLstmMultiCropModel, ModelType.XgazeNonLstmMultiCropSepModel]:
        test_set = GazeMulticropDataset(dataset_path=data_dir,
                                        keys_to_use=datastore[sub_folder_use],
                                        sub_folder=sub_folder_use,
                                        cropsizes=cropsizes,
                                        transform=trans,
                                        is_shuffle=False,
                                        is_load_label=False)

    elif model_type == ModelType.XgazeStaticAdvModel:
        test_set = GazeDatasetWithId(dataset_path=data_dir,
                                     keys_to_use=datastore[sub_folder_use],
                                     sub_folder=sub_folder_use,
                                     sc_target=sc_target,
                                     transform=trans,
                                     is_shuffle=False,
                                     is_load_label=False)
    elif model_type == ModelType.XgazeStaticCLModel:
        test_set = GazeDatasetCL(dataset_path=data_dir,
                                 keys_to_use=datastore[sub_folder_use],
                                 sub_folder=sub_folder_use,
                                 sc_target=sc_target,
                                 transform=trans,
                                 is_shuffle=False,
                                 is_load_label=False)
    elif model_type in [
            ModelType.XgazeStaticWarpModel, ModelType.XgazeStaticWarp3Model, ModelType.XgazeStaticWarp4Model,
            ModelType.XgazeStaticWarp2Model, ModelType.XgazeStaticWarp5Model, ModelType.XgazeStaticWarp6Model
    ]:
        test_set = GazeDatasetWithWarping(dataset_path=data_dir,
                                          yaw_warping_angle=yaw_warping_angle,
                                          pitch_warping_angle=pitch_warping_angle,
                                          keys_to_use=datastore[sub_folder_use],
                                          sub_folder=sub_folder_use,
                                          sc_target=sc_target,
                                          transform=trans,
                                          is_shuffle=False,
                                          is_load_label=False)

    elif model_type in [
            ModelType.XgazeStaticPairModel, ModelType.XgazeStaticPair2Model, ModelType.XgazeStaticPinballPairModel
    ]:
        bin_size = kwargs['bin_size']
        ignore_nbr_cnt = kwargs['ignore_nbr_cnt']
        sample_nbr_cnt = kwargs['sample_nbr_cnt']
        ignore_same_bucket = kwargs['ignore_same_bucket']
        test_set = PairedDataLoader(dataset_path=data_dir,
                                    keys_to_use=datastore[sub_folder_use],
                                    sub_folder=sub_folder_use,
                                    transform=trans,
                                    yaw_bin_size=bin_size,
                                    pitch_bin_size=bin_size,
                                    ignore_nbr_cnt=ignore_nbr_cnt,
                                    sample_nbr_cnt=sample_nbr_cnt,
                                    ignore_same_bucket=ignore_same_bucket,
                                    is_shuffle=False,
                                    is_load_label=False)
    return test_set
