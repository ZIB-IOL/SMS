# ===========================================================================
# Project:      Sparse Model Soups: A Recipe for Improved Pruning via Model Averaging - IOL Lab @ ZIB
# Paper:        arxiv.org/abs/2306.16788
# File:         models/imagenet.py
# Description:  ImageNet Models
# ===========================================================================

import torchvision


def ResNet50():
    return torchvision.models.resnet50(pretrained=False)
