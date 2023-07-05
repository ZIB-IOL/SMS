# ===========================================================================
# Project:      Sparse Model Soups: A Recipe for Improved Pruning via Model Averaging - IOL Lab @ ZIB
# Paper:        arxiv.org/abs/2306.16788
# File:         models/cifar10.py
# Description:  CIFAR-10 Models
# ===========================================================================


import torch.nn as nn
import torch.nn.functional as F

from utilities.utilities import Utilities as Utils


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, bias=True)
        self.conv2 = nn.Conv2d(32, 128, 3, 1, bias=True)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.avg = nn.AvgPool2d(kernel_size=1, stride=1)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 10, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    @staticmethod
    def get_permutation_spec():
        conv = lambda name, p_in, p_out, bias=True: {f"{name}.weight": (p_out, p_in, None, None,),
                                                     f"{name}.bias": (p_out,)} if bias else {
            f"{name}.weight": (p_out, p_in, None, None,)}
        dense = lambda name, p_in, p_out, bias=True: {f"{name}.weight": (p_out, p_in),
                                                      f"{name}.bias": (p_out,)} if bias else {
            f"{name}.weight": (p_out, p_in)}
        return Utils.permutation_spec_from_axes_to_perm({
            **conv("conv1", None, "P_bg0"),
            **conv("conv2", "P_bg0", "P_bg1", False),
            **dense("fc1", "P_bg1", "P_bg2"),
            **dense("fc2", "P_bg2", None, True),
        })


def ResNet56():
    class ResNet(nn.Module):
        # Proper implementation of ResNet, taken from https://github.com/JJGO/shrinkbench/blob/master/models/cifar_resnet.py
        def __init__(self, block, num_blocks, num_classes=10):
            super(ResNet, self).__init__()
            self.in_planes = 16

            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
            self.linear = nn.Linear(64, num_classes)
            # self.linear.is_classifier = True    # So layer is not pruned

        def _make_layer(self, block, planes, num_blocks, stride):
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion

            return nn.Sequential(*layers)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = F.avg_pool2d(out, out.size()[3])
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out

    class LambdaLayer(nn.Module):
        def __init__(self, lambd):
            super(LambdaLayer, self).__init__()
            self.lambd = lambd

        def forward(self, x):
            return self.lambd(x)

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1, option='A'):
            super(BasicBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != planes:
                if option == 'A':
                    """
                    For CIFAR10 ResNet paper uses option A.
                    """
                    self.shortcut = LambdaLayer(lambda x:
                                                F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4),
                                                      "constant", 0))
                elif option == 'B':
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(self.expansion * planes)
                    )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out

    model = ResNet(BasicBlock, [9, 9, 9], num_classes=10)
    return model


def ResNet18():
    # Based on https://github.com/charlieokonomiyaki/pytorch-resnet18-cifar10/blob/master/models/resnet.py
    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1):
            super(BasicBlock, self).__init__()
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out

    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, in_planes, planes, stride=1):
            super(Bottleneck, self).__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, self.expansion *
                                   planes, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(self.expansion * planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out

    class ResNet(nn.Module):
        def __init__(self, block, num_blocks, num_classes=10):
            super(ResNet, self).__init__()
            self.in_planes = 64

            self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.linear = nn.Linear(512 * block.expansion, num_classes)

        def _make_layer(self, block, planes, num_blocks, stride):
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out

    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
    return model


def VGG16():
    return VGG(vgg_name='VGG16')


class VGG(nn.Module):
    # Adapted from https://github.com/jaeho-lee/layer-adaptive-sparsity/blob/main/tools/models/vgg.py
    def __init__(self, vgg_name, use_bn=True):
        cfg = {
            'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                      'M'],
        }

        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], use_bn)
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, use_bn):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1)]
                if use_bn:
                    layers += [nn.BatchNorm2d(x)]
                layers += [nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class WideResNet20(nn.Module):
    # WideResNet implementation but with widen_factor=2 and depth=22 instead of 10 and 28 respectively.
    # In Repo of Git-Rebasin, this is referred to as ResNet20
    def __init__(self, depth=22, widen_factor=10, dropout_rate=0.3, num_classes=10):
        super(WideResNet20, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k]

        class wide_basic(nn.Module):
            def __init__(self, in_planes, planes, dropout_rate, stride=1):
                super(wide_basic, self).__init__()
                self.bn1 = nn.BatchNorm2d(in_planes)
                self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)

                self.dropout = nn.Dropout(p=dropout_rate)
                self.bn2 = nn.BatchNorm2d(planes)
                self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

                self.shortcut = nn.Sequential()
                if stride != 1 or in_planes != planes:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
                    )

            def forward(self, x):
                out = self.dropout(self.conv1(F.relu(self.bn1(x))))
                out = self.conv2(F.relu(self.bn2(out)))
                out += self.shortcut(x)

                return out

        self.conv1 = self.conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def conv3x3(self, in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

    @staticmethod
    def get_permutation_spec():
        conv = lambda name, p_in, p_out, bias=True: {f"{name}.weight": (p_out, p_in, None, None,),
                                                     f"{name}.bias": (p_out,)} if bias else {
            f"{name}.weight": (p_out, p_in, None, None,)}
        norm = lambda name, p: {f"{name}.weight": (p,), f"{name}.bias": (p,)}

        dense = lambda name, p_in, p_out, bias=True: {f"{name}.weight": (p_out, p_in),
                                                      f"{name}.bias": (p_out,)} if bias else {
            f"{name}.weight": (p_out, p_in)}

        # This is for easy blocks that use a residual connection, without any change in the number of channels.
        easyblock = lambda name, p: {
            **norm(f"{name}.bn1", p),
            **conv(f"{name}.conv1", p, f"P_{name}_inner"),
            **norm(f"{name}.bn2", f"P_{name}_inner"),
            **conv(f"{name}.conv2", f"P_{name}_inner", p),
        }

        # This is for blocks that use a residual connection, but change the number of channels via a Conv.
        shortcutblock = lambda name, p_in, p_out: {
            **norm(f"{name}.bn1", p_in),
            **conv(f"{name}.conv1", p_in, f"P_{name}_inner"),
            **norm(f"{name}.bn2", f"P_{name}_inner"),
            **conv(f"{name}.conv2", f"P_{name}_inner", p_out),
            **conv(f"{name}.shortcut.0", p_in, p_out),
            # **norm(f"{name}.shortcut.1", p_out),   # Removed this since not occuring in state dict
        }

        return Utils.permutation_spec_from_axes_to_perm({
            **conv("conv1", None, "P_bg0"),
            #
            **shortcutblock("layer1.0", "P_bg0", "P_bg1"),
            **easyblock("layer1.1", "P_bg1", ),
            **easyblock("layer1.2", "P_bg1"),
            # **easyblock("layer1.3", "P_bg1"),

            **shortcutblock("layer2.0", "P_bg1", "P_bg2"),
            **easyblock("layer2.1", "P_bg2", ),
            **easyblock("layer2.2", "P_bg2"),
            # **easyblock("layer2.3", "P_bg2"),

            **shortcutblock("layer3.0", "P_bg2", "P_bg3"),
            **easyblock("layer3.1", "P_bg3", ),
            **easyblock("layer3.2", "P_bg3"),
            # **easyblock("layer3.3", "P_bg3"),

            **norm("bn1", "P_bg3"),

            **dense("linear", "P_bg3", None),

        })
