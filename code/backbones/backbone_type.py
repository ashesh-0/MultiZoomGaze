from core.enum import Enum


class BackboneType(Enum):
    Resnet18 = 0
    Squeezenet = 1
    DenseNet = 2
    ShuffleNet = 3
    MobileNet = 4
    HardNet = 5
    EfficientNet = 6
