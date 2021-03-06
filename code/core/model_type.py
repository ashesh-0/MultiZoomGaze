from core.enum import Enum


class ModelType(Enum):
    LSTM = 1
    StaticModel = 2
    StaticXyzModel = 3
    StaticBackwardModel = 4
    LSTMBackward = 5
    StaticSinModel = 6
    StaticCosModel = 7
    StaticSinCosModel = 8
    StaticSinCosRegModel = 9
    SinCosRegModel = 10
    StaticSinCosAllRegModel = 11
    StaticSinCosRegEyeDropoutModel = 12
    StaticSinCosRegKpDropoutModel = 13
    StaticSinCosRegEyeBlurModel = 14
    StaticSinCosRegSamplSelnModel = 15
    StaticWeightedSinCosRegModel = 16
    StaticWeightedMseSinCosModel = 17
    StaticSinCosRegBlurModel = 18
    StaticSinCosRegLeanModel = 19
    StaticSinCosRegDiscrimModel = 20
    StaticSinCosRegDiscrimAccurateIdModel = 21
    StaticSinCosRegDiscrimSymmetricModel = 22
    StaticCoarseToFineMPIIModel = 23
    StaticCoarseToFine2EyeMPIIModel = 24
    StaticCoarseToFineGaze360Model = 25
    StaticSinCosRegAttentionEyeModel = 26
    StaticSinCosEyeModel = 27
    StaticCoarseToFineGaze360PinballModel = 28
    StaticSinCosRegHavingEyesModel = 29
    SpatialTransformerModel = 30
    MultiScaleModel = 31
    SinCosRegLstmScaleModel = 32
    SinCosRegLstmSelectiveScaleModel = 33
    SinCosModel = 34
    StaticMultiSinCosRegModel = 35
    StaticCoarseToFineMPIIFaceModel = 36
    StaticRTGENEModel = 37
    RTGENEModel = 38
    Gaze360MultiCropModel = 39
    NonLstmSinCosRegModel = 40
    NonLstmSinCosModel = 41
    NonLstmMultiCropModel = 42
    NonLstmRTGENEModel = 43
    NonLstmSinCosRandomModel = 44
    XgazeStaticModel = 45
    XgazeNonLstmMultiCropModel = 46
    XgazeStaticSCModel = 47
    XgazeStaticSepModel = 48
    XgazeStaticAdvModel = 49
    XgazeNonLstmMultiCropSepModel = 50
    XgazeStaticCLModel = 51
    XgazeNonLstmMultiCropMergeModel = 52
    XgazeMultiCropAdditiveModel = 53
    XgazeStaticWarpModel = 54
    XgazeStaticWarp3Model = 55
    XgazeStaticWarp4Model = 56
    XgazeStaticWarp2Model = 57
    XgazeStaticWarp5Model = 58
    XgazeStaticWarp6Model = 59
    XgazeStaticPair2Model = 60
    XgazeStaticPairModel = 61
    XgazeStaticPairPretrainedModel = 62
    Gaze360StaticPairModel = 63
    XgazeStaticPinballPairModel = 64
    Gaze360CutMixModel = 65
    Gaze360EqCutMixModel = 66
    Gaze360HybridFaceCutMixModel = 67
    Gaze360HalfSizedModel = 68
    Gaze360MultiCropEqCutMixModel = 69
    Gaze360LazyAggregationModel = 70
