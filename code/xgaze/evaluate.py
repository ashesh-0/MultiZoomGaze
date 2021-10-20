import numpy as np
import torch
from tqdm import tqdm

from xgaze.xgaze_dataloader import get_testset


def generate_test_csv(model, data_dir, model_type, cropsizes=None, kfold_id=None, batch_size=50, workers=4):
    """
        Test the pre-treained model on the whole test set. Note there is no label released to public, you can
        only save the predicted results. You then need to submit the test resutls to our evaluation website to
        get the final gaze estimation error.
        """
    print('We are now doing the final test')
    model.eval()
    dataset = get_testset(data_dir, model_type, cropsizes=cropsizes)
    num_test = len(dataset)
    pred_gaze_all = np.zeros((num_test, 2))
    save_index = 0
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=workers,
                                         pin_memory=True)

    print('Testing on ', num_test, ' samples')
    for i, (input_img) in tqdm(enumerate(loader)):
        input_var = torch.autograd.Variable(input_img.float().cuda())
        pred_gaze, _ = model(input_var)
        pred_gaze_all[save_index:save_index + batch_size, :] = pred_gaze.cpu().data.numpy()
        save_index += input_var.size(0)

    if save_index != num_test:
        print('the test samples save_index ', save_index, ' is not equal to the whole test set ', num_test)

    print('Tested on : ', pred_gaze_all.shape[0], ' samples')
    # We predict [yaw,pitch], they expect [pitch,yaw] opposite.
    pred_gaze_all = pred_gaze_all[:, ::-1]
    if kfold_id is None:
        fpath = 'within_eva_results.txt'
    else:
        fpath = f'within_eva_results_K:{kfold_id}.txt'

    np.savetxt(fpath, pred_gaze_all, delimiter=',')

    print('Written to', fpath)
