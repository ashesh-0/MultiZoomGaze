from backbones.interface import (get_hardnet, get_hardnet_cnn, get_hardnet_head, get_mobilenet, get_mobilenet_cnn,
                                 get_mobilenet_head, get_resnet, get_resnet_cnn, get_resnet_head, get_shufflenet,
                                 get_shufflenet_cnn, get_shufflenet_head, get_sqeezenet, get_sqeezenet_cnn,
                                 get_sqeezenet_head)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_hardnet_division():
    cnn = get_hardnet_cnn()
    head = get_hardnet_head(output_dim=256)
    model = get_hardnet(output_dim=256)
    assert count_params(model) == (count_params(head) + count_params(cnn))


def test_resnet_division():
    cnn = get_resnet_cnn()
    head = get_resnet_head(output_dim=256)
    model = get_resnet(output_dim=256)
    assert count_params(model) == (count_params(head) + count_params(cnn))


def test_mobilenet_division():
    cnn = get_mobilenet_cnn()
    head = get_mobilenet_head(output_dim=256)
    model = get_mobilenet(output_dim=256)
    assert count_params(model) == (count_params(head) + count_params(cnn))


def test_shufflenet_division():
    cnn = get_shufflenet_cnn()
    head = get_shufflenet_head(output_dim=256)
    model = get_shufflenet(output_dim=256)
    assert count_params(model) == (count_params(head) + count_params(cnn))


def test_sqeezenet_division():
    cnn = get_sqeezenet_cnn()
    head = get_sqeezenet_head(output_dim=256)
    model = get_sqeezenet(output_dim=256)
    assert count_params(model) == (count_params(head) + count_params(cnn))
