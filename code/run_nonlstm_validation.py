import torch
import torchvision.transforms as transforms

from core.prediction_utils import get_prediction_with_uncertainty
from core.train_utils import checkpoint_params
from sinecosine_model.data_loader_static_sinecosine import ImageLoaderStaticSineCosine
from sinecosine_model.static_sinecosine_model import GazeStaticSineAndCosineModel
from sinecosine_model.train_utils import compute_angular_error_sine_and_cosine

# test_file = "/home/u4421059/validation.txt"
# source_path = "/home/u4421059/data/imgs"
# checkpoint_test = "/home/u4421059/checkpoints/model_best_gaze360_static_TYPE:9_fc1:None_fc2:256_bkb:4_imsz:224_bsz:64_lr:0.0001_v:master_1.pth.tar"
test_file = 'validation.txt'
source_path = '/tmp2/ashesh/gaze360_data/imgs/'
checkpoint_test = "/home/ashesh/twcc/checkpoints/model_best_gaze360_static_TYPE:9_fc1:None_fc2:256_bkb:4_imsz:224_bsz:64_lr:0.0001_v:master_1.pth.tar"
params = checkpoint_params(checkpoint_test)
centercrop = int(params.get('centercrop', 0))
backbone_type = int(params['bkb'])
# centercrop_list = [224]

batch_size = 32
num_workers = 2

image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# criterion = SinCosLoss().cuda()

model_v = GazeStaticSineAndCosineModel(fc1=None, fc2=int(params['fc2']), backbone_type=backbone_type)
# # device = torch.device("cuda:1")
# # model_v.to(device)
# # model = model_v

model = torch.nn.DataParallel(model_v).cuda()
_ = model.cuda()

checkpoint = torch.load(checkpoint_test)
_ = model.load_state_dict(checkpoint['state_dict'])

checkpoint['epoch']

if centercrop > 0:
    print("Using Centercrop", centercrop)
    img_loader = ImageLoaderStaticSineCosine(source_path, test_file,
                                             transforms.Compose([
                                                 transforms.Resize((224, 224)),
                                                 transforms.CenterCrop(centercrop),
                                                 transforms.Resize((224, 224)),
                                                 transforms.ToTensor(),
                                                 image_normalize,
                                             ]))
else:
    img_loader = ImageLoaderStaticSineCosine(source_path, test_file,
                                             transforms.Compose([
                                                 transforms.Resize((224, 224)),
                                                 transforms.ToTensor(),
                                                 image_normalize,
                                             ]))

(prediction, actual, uncertainty) = get_prediction_with_uncertainty(model, img_loader, num_workers=num_workers)
print(len(img_loader))
print(img_loader[0][0][:5, :5, 0])
print(img_loader[0][1])
# print(prediction[:5, :])
print(compute_angular_error_sine_and_cosine(torch.Tensor(prediction), torch.Tensor(actual)))
