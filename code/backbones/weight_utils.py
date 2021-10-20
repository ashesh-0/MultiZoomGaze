import torch


def load_backbone_weights(model, checkpoint_fpath):
    prefix = 'module.base_model.'
    bm_dict = {}
    checkpoint_dict = torch.load(checkpoint_fpath)
    for key, value in checkpoint_dict['state_dict'].items():
        if key[:len(prefix)] == prefix:
            bm_dict[key[len(prefix):]] = value

    model.base_model.load_state_dict(bm_dict)
    print(f'[BackboneWeights] Loading base_model weights from {checkpoint_fpath}')


def freeze_backbone_weights(model, freeze_layer_idx):
    assert freeze_layer_idx > 0
    for i, nparam in enumerate(model.base_model.named_parameters()):
        name, param = nparam
        if i >= freeze_layer_idx:
            print('[BackboneWeights] Freezing layers before', name)
            break
        param.requires_grad = False
