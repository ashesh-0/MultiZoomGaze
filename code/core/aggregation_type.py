from core.enum import Enum


class AggregationType(Enum):
    MAX = 0
    MEAN = 1
    SPATIAL_MAX = 2
    SPATIAL_ATTENTION = 3
    SPATIAL_ATTENTION_ALIGNED = 4
    SPATIAL_ATTENTION_V2 = 5
    ATTENTION = 6
    SPATIAL_MEAN = 7
