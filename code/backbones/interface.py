import torch
import torch.nn as nn

from backbones.backbone_type import BackboneType
from backbones.resnet import ViewDense, resnet18
from efficientnet_pytorch import EfficientNet


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.shape[0], ) + self.shape)


def get_efficientnet(pretrained=True, output_dim=None):
    assert output_dim is not None
    model = EfficientNet.from_pretrained('efficientnet-b2')
    model._dropout = nn.Identity()
    return nn.Sequential(model, nn.Linear(1000, output_dim))


def get_efficientnet_cnn(pretrained=True):
    model = EfficientNet.from_pretrained('efficientnet-b2', include_top=False)
    model._avg_pooling = nn.Identity()
    return model


def get_efficientnet_head(pretrained=True, output_dim=None):
    assert output_dim is not None
    model = EfficientNet.from_pretrained('efficientnet-b2')

    return nn.Sequential(
        model._avg_pooling,
        ViewDense(),
        model._dropout,
        model._fc,
        model._swish,
        nn.Linear(1000, output_dim),
    )


def get_resnet(pretrained=True, output_dim=None):
    assert output_dim is not None
    model = resnet18(pretrained=pretrained)
    model.fc2 = nn.Linear(1000, output_dim)
    return model


def get_resnet_cnn(pretrained=True):
    base_model = get_resnet(pretrained=pretrained, output_dim=256)
    base_model.avgpool = nn.Identity()
    base_model.view = nn.Identity()
    base_model.fc1 = nn.Identity()
    base_model.fc1_activation = nn.Identity()
    base_model.fc2 = nn.Identity()
    return base_model


def get_resnet_head(pretrained=True, output_dim=None):
    assert output_dim is not None
    return nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        ViewDense(),
        nn.Linear(512, 1000),
        nn.ReLU(),
        nn.Linear(1000, output_dim),
    )


class Flatten(nn.Module):
    def forward(self, inp):
        return inp.view((inp.shape[0], inp.shape[1]))


def get_sqeezenet(pretrained=True, output_dim=None):
    assert output_dim is not None
    model = torch.hub.load('pytorch/vision:v0.5.0', 'squeezenet1_0', pretrained=pretrained)
    return nn.Sequential(model, Flatten(), nn.Linear(1000, output_dim))


def get_sqeezenet_cnn(pretrained=True):
    model = torch.hub.load('pytorch/vision:v0.5.0', 'squeezenet1_0', pretrained=pretrained)
    classifier = list(model.classifier.children())[:-1]
    model.classifier = nn.Sequential(*classifier)
    return nn.Sequential(model, Reshape(1000, 13, 13))


def get_sqeezenet_head(pretrained=True, output_dim=None):
    return nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), ViewDense(), nn.Linear(1000, output_dim))


def get_hardnet(pretrained=True, output_dim=None, version='hardnet68'):
    assert output_dim is not None
    model = torch.hub.load('PingoLH/Pytorch-HarDNet', version, pretrained=True)
    return nn.Sequential(model, nn.Linear(1000, output_dim))


def get_hardnet_cnn(pretrained=True, version='hardnet68'):
    model = torch.hub.load('PingoLH/Pytorch-HarDNet', version, pretrained=True)
    model.base = nn.Sequential(*list(model.base.children())[:-1])
    return model


def get_hardnet_head(pretrained=True, output_dim=None, version='hardnet68'):
    assert output_dim is not None
    model = torch.hub.load('PingoLH/Pytorch-HarDNet', version, pretrained=True)
    head = [
        list(model.base.children())[-1],
        nn.Linear(1000, output_dim),
    ]

    return nn.Sequential(*head)


def get_densenet(pretrained=True, output_dim=None):
    assert output_dim is not None
    model = torch.hub.load('pytorch/vision:v0.5.0', 'densenet121', pretrained=pretrained)
    model.classifier = nn.Linear(1024, output_dim)
    return model


def get_shufflenet(pretrained=True, output_dim=None):
    assert output_dim is not None
    model = torch.hub.load('pytorch/vision:v0.5.0', 'shufflenet_v2_x1_0', pretrained=pretrained)
    model.fc = nn.Linear(1024, output_dim)
    return model


def get_shufflenet_cnn(pretrained=True):
    model = torch.hub.load('pytorch/vision:v0.5.0', 'shufflenet_v2_x1_0', pretrained=pretrained)
    return nn.Sequential(*list(model.children())[:-1])


def get_shufflenet_head(pretrained=True, output_dim=None):
    assert output_dim is not None
    return nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), ViewDense(), nn.Linear(1024, output_dim))


def get_mobilenet(pretrained=True, output_dim=None):
    assert output_dim is not None
    model = torch.hub.load('pytorch/vision:v0.5.0', 'mobilenet_v2', pretrained=pretrained)
    model.classifier = nn.Identity()
    return nn.Sequential(model, nn.Linear(1280, output_dim))


def get_mobilenet_cnn(pretrained=True):
    model = torch.hub.load('pytorch/vision:v0.5.0', 'mobilenet_v2', pretrained=pretrained)
    return model.features


def get_mobilenet_head(pretrained=True, output_dim=None):
    assert output_dim is not None
    return nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), ViewDense(), nn.Linear(1280, output_dim))


def get_backbone(backbone_type, **kwargs):
    print(f'[Backbone] Using {BackboneType.name(backbone_type)} Pretrained:{kwargs["pretrained"]}')
    if backbone_type == BackboneType.Resnet18:
        return get_resnet(**kwargs)
    elif backbone_type == BackboneType.Squeezenet:
        return get_sqeezenet(**kwargs)

    elif backbone_type == BackboneType.DenseNet:
        return get_densenet(**kwargs)
    elif backbone_type == BackboneType.ShuffleNet:
        return get_shufflenet(**kwargs)
    elif backbone_type == BackboneType.MobileNet:
        return get_mobilenet(**kwargs)
    elif backbone_type == BackboneType.HardNet:
        return get_hardnet(**kwargs)
    elif backbone_type == BackboneType.EfficientNet:
        return get_efficientnet(**kwargs)
    else:
        raise Exception(f'Invalid backbone_type: {backbone_type}')


def get_backbone_cnn(backbone_type, **kwargs):
    print(f'[BackboneCNN] Using {BackboneType.name(backbone_type)} Pretrained:{kwargs["pretrained"]}')
    if backbone_type == BackboneType.Resnet18:
        return get_resnet_cnn(**kwargs)
    elif backbone_type == BackboneType.Squeezenet:
        return get_sqeezenet_cnn(**kwargs)
    elif backbone_type == BackboneType.ShuffleNet:
        return get_shufflenet_cnn(**kwargs)
    elif backbone_type == BackboneType.MobileNet:
        return get_mobilenet_cnn(**kwargs)
    elif backbone_type == BackboneType.HardNet:
        return get_hardnet_cnn(**kwargs)
    elif backbone_type == BackboneType.EfficientNet:
        return get_efficientnet_cnn(**kwargs)
    else:
        raise Exception(f'Invalid backbone_type: {backbone_type}')


def get_backbone_head(backbone_type, **kwargs):
    print(f'[BackboneHead] Using {BackboneType.name(backbone_type)} Pretrained:{kwargs["pretrained"]}')
    if backbone_type == BackboneType.Resnet18:
        return get_resnet_head(**kwargs)
    elif backbone_type == BackboneType.Squeezenet:
        return get_sqeezenet_head(**kwargs)
    elif backbone_type == BackboneType.ShuffleNet:
        return get_shufflenet_head(**kwargs)
    elif backbone_type == BackboneType.MobileNet:
        return get_mobilenet_head(**kwargs)
    elif backbone_type == BackboneType.HardNet:
        return get_hardnet_head(**kwargs)
    elif backbone_type == BackboneType.EfficientNet:
        return get_efficientnet_head(**kwargs)
    else:
        raise Exception(f'Invalid backbone_type: {backbone_type}')
