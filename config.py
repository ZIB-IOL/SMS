# ===========================================================================
# Project:      Sparse Model Soups: A Recipe for Improved Pruning via Model Averaging - IOL Lab @ ZIB
# Paper:        arxiv.org/abs/2306.16788
# File:         config.py
# Description:  Datasets, Normalization and Transforms
# ===========================================================================

import numpy as np
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor


class CIFARCORRUPT(Dataset):
    # CIFAR10CORRUPT and CIFAR100CORRUPT are the same, only the root changes
    def __init__(self, root, corruption='gaussian_noise', severity=3, transform=ToTensor()):
        self.root = root
        self.corruption = corruption  # e.g. 'gaussian_noise'
        self.severity = severity  # in [1, 2, 3, 4, 5]
        self.transform = transform
        data = np.load(f'{root}/{corruption}.npy')
        self.labels = np.load(f'{root}/labels.npy')

        # Only load images with the specified severity level
        start_index = (severity - 1) * 10000
        end_index = severity * 10000
        self.data = data[start_index:end_index]
        self.labels = self.labels[start_index:end_index]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label


means = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
    'imagenet': (0.485, 0.456, 0.406),
    'tinyimagenet': (0.485, 0.456, 0.406),
}

stds = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
    'imagenet': (0.229, 0.224, 0.225),
    'tinyimagenet': (0.229, 0.224, 0.225),
}

datasetDict = {  # Links dataset names to actual torch datasets
    'mnist': getattr(torchvision.datasets, 'MNIST'),
    'cifar10': getattr(torchvision.datasets, 'CIFAR10'),
    'fashionMNIST': getattr(torchvision.datasets, 'FashionMNIST'),
    'SVHN': getattr(torchvision.datasets, 'SVHN'),  # This needs scipy
    'STL10': getattr(torchvision.datasets, 'STL10'),
    'cifar100': getattr(torchvision.datasets, 'CIFAR100'),
    'imagenet': getattr(torchvision.datasets, 'ImageNet'),
    'tinyimagenet': getattr(torchvision.datasets, 'ImageFolder'),
    'CIFAR10CORRUPT': CIFARCORRUPT,
    'CIFAR100CORRUPT': CIFARCORRUPT,
}

trainTransformDict = {  # Links dataset names to train dataset transformers
    'mnist': transforms.Compose([transforms.ToTensor()]),
    'cifar10': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['cifar10'], std=stds['cifar10']), ]),
    'cifar100': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['cifar100'], std=stds['cifar100']), ]),
    'imagenet': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['imagenet'], std=stds['imagenet']), ]),
    'tinyimagenet': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['tinyimagenet'], std=stds['tinyimagenet']), ]),
}
testTransformDict = {  # Links dataset names to test dataset transformers
    'mnist': transforms.Compose([transforms.ToTensor()]),
    'cifar10': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=means['cifar10'], std=stds['cifar10']), ]),
    'cifar100': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=means['cifar100'], std=stds['cifar100']), ]),
    'imagenet': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['imagenet'], std=stds['imagenet']), ]),
    'tinyimagenet': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['tinyimagenet'], std=stds['tinyimagenet']), ]),
}
