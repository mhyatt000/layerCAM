'''finds layer for LayerCAMs'''

#TODO: might not need the imports
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from .imagenet import *


def find_resnet_layer(arch, name='layer4'):
    """Find resnet layer to calculate GradCAM and GradCAM++
    Args:
        arch: default torchvision densenet models
        name (str): the name of layer with its hierarchical information. please refer to usages below.
            name = 'conv1'
            name = 'layer1'
            name = 'layer1_basicblock0'
            name = 'layer1_basicblock0_relu'
            name = 'layer1_bottleneck0'
            name = 'layer1_bottleneck0_conv1'
            name = 'layer1_bottleneck0_downsample'
            name = 'layer1_bottleneck0_downsample_0'
            name = 'avgpool'
            name = 'fc'
    Return:
        layer: found layer. this layer will be hooked to get forward/backward pass information.
    """

    if "layer" in name:

        hierarchy = name.split("_")
        num = int(hierarchy[0].lstrip("layer"))

        try:
            layer = getattr(arch,f'layer{num}')
        except:
            raise ValueError(f"unknown layer : {name}")

        if len(hierarchy) >= 2:
            bottleneck_num = int(hierarchy[1].lower().lstrip("bottleneck").lstrip("basicblock"))
            layer = layer[bottleneck_num]

        if len(hierarchy) >= 3:
            layer = layer._modules[hierarchy[2]]

        if len(hierarchy) == 4:
            layer = layer._modules[hierarchy[3]]

    else:
        layer = arch._modules[name]

    return layer


#TODO: use negative indexing to clean up the code if needed

def find_densenet_layer(arch, name='features'):
    """Find densenet layer to calculate GradCAM and GradCAM++
    Args:
        arch: default torchvision densenet models
        name (str): the name of layer with its hierarchical information. please refer to usages below.
            name = 'features'
            name = 'features_transition1'
            name = 'features_transition1_norm'
            name = 'features_denseblock2_denselayer12'
            name = 'features_denseblock2_denselayer12_norm1'
            name = 'features_denseblock2_denselayer12_norm1'
            name = 'classifier'
    Return:
        layer: found layer. this layer will be hooked to get forward/backward pass information.
    """

    hierarchy = name.split("_")
    layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        layer = layer._modules[hierarchy[1]]

    if len(hierarchy) >= 3:
        layer = layer._modules[hierarchy[2]]

    if len(hierarchy) == 4:
        layer = layer._modules[hierarchy[3]]

    return layer


def find_vgg_layer(arch, name='features'):
    """Find vgg layer to calculate GradCAM and GradCAM++
    Args:
        arch: default torchvision densenet models
        name (str): the name of layer with its hierarchical information. please refer to usages below.
            name = 'features'
            name = 'features_42'
            name = 'classifier'
            name = 'classifier_0'
    Return:
        layer: found layer. this layer will be hooked to get forward/backward pass information.
    """

    hierarchy = name.split("_")

    if len(hierarchy) >= 1:
        layer = arch.features

    if len(hierarchy) == 2:
        layer = layer[int(hierarchy[1])]

    return layer


def find_alexnet_layer(arch, name='features_29'):
    """Find alexnet layer to calculate GradCAM and GradCAM++
    Args:
        arch: default torchvision densenet models
        name (str): the name of layer with its hierarchical information. please refer to usages below.
            name = 'features'
            name = 'features_0'
            name = 'classifier'
            name = 'classifier_0'
    Return:
        layer: found layer. this layer will be hooked to get forward/backward pass information.
    """

    hierarchy = name.split("_")

    if len(hierarchy) >= 1:
        layer = arch.features

    if len(hierarchy) == 2:
        layer = layer[int(hierarchy[1])]

    return layer


def find_squeezenet_layer(arch, name='features'):
    """Find squeezenet layer to calculate GradCAM and GradCAM++
    Args:
        - **arch - **: default torchvision densenet models
        - **name (str) - **: the name of layer with its hierarchical information. please refer to usages below.
            name = 'features_12'
            name = 'features_12_expand3x3'
            name = 'features_12_expand3x3_activation'
    Return:
        layer: found layer. this layer will be hooked to get forward/backward pass information.
    """

    hierarchy = name.split("_")
    layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        layer = layer._modules[hierarchy[1]]

    if len(hierarchy) == 3:
        layer = layer._modules[hierarchy[2]]

    elif len(hierarchy) == 4:
        layer = layer._modules[hierarchy[2] + "_" + hierarchy[3]]

    return layer


def find_googlenet_layer(arch, name='features'):
    """Find squeezenet layer to calculate GradCAM and GradCAM++
    Args:
        - **arch - **: default torchvision googlenet models
        - **name (str) - **: the name of layer with its hierarchical information. please refer to usages below.
            name = 'inception5b'
    Return:
        layer: found layer. this layer will be hooked to get forward/backward pass information.
    """

    hierarchy = name.split("_")
    layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        layer = layer._modules[hierarchy[1]]

    if len(hierarchy) == 3:
        layer = layer._modules[hierarchy[2]]

    elif len(hierarchy) == 4:
        layer = layer._modules[hierarchy[2] + "_" + hierarchy[3]]

    return layer


def find_mobilenet_layer(arch, name='features'):
    """Find mobilenet layer to calculate GradCAM and GradCAM++
    Args:
        - **arch - **: default torchvision googlenet models
        - **name (str) - **: the name of layer with its hierarchical information. please refer to usages below.
            name = 'features'
    Return:
        layer: found layer. this layer will be hooked to get forward/backward pass information.
    """

    hierarchy = name.split("_")
    layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        layer = layer._modules[hierarchy[1]]

    if len(hierarchy) == 3:
        layer = layer._modules[hierarchy[2]]

    elif len(hierarchy) == 4:
        layer = layer._modules[hierarchy[2] + "_" + hierarchy[3]]

    return layer


def find_shufflenet_layer(arch, name='features'):
    """Find mobilenet layer to calculate GradCAM and GradCAM++
    Args:
        - **arch - **: default torchvision googlenet models
        - **name (str) - **: the name of layer with its hierarchical information. please refer to usages below.
            name = 'conv5'
    Return:
        layer: found layer. this layer will be hooked to get forward/backward pass information.
    """

    hierarchy = name.split("_")
    layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        layer = layer._modules[hierarchy[1]]

    if len(hierarchy) == 3:
        layer = layer._modules[hierarchy[2]]

    elif len(hierarchy) == 4:
        layer = layer._modules[hierarchy[2] + "_" + hierarchy[3]]

    return layer


def find_layer(arch, name):
    """Find target layer to calculate CAM.
        : Args:
            - **arch - **: Self-defined architecture.
            - **name - ** (str): Name of target class.
        : Return:
            - **layer - **: Found layer. This layer will be hooked to get forward/backward pass information.
    """

    print('generic')

    try:
        layer = arch._modules[name.split("_")]
    except:
        raise Exception("Invalid target layer name.")
    return layer

