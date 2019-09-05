import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch import load
from copy import deepcopy

from components import branches
from components.shallow_cam import ShallowCAM
import torchvision.models.resnet
__all__ = ['resnet50',]


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.is_last = is_last

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if not self.is_last:
            out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, class_num=1000, last_stride=2, is_for_test=False, norm=False):
        self.inplanes = 64
        self.block = block
        self.class_num = class_num
        self.is_for_test = is_for_test
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride, is_last=norm)
        self.avgpool = nn.AvgPool2d(kernel_size=(7, 7), stride=1)
        if self.is_for_test is False:
            self.fc = nn.Linear(512 * block.expansion, class_num)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, is_last=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, is_last=is_last))

        return nn.Sequential(*layers)

    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.maxpool(x)

    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)

        # if self.is_for_test:
        #     return x
        # x = self.fc(x)
        # return x


    # def get_param(self, lr):
    #     new_param = self.fc.parameters()
    #     # return new_param
    #     new_param_id = [id(p) for p in new_param]
    #     finetuned_params = []
    #     for p in self.parameters():
    #         if id(p) not in new_param_id:
    #             finetuned_params.append(p)
    #     return [{'params': new_param, 'lr': lr},
    #             {'params': finetuned_params, 'lr': 1e-1 * lr}]


class ResNetCommonBranch(nn.Module):

    def __init__(self, owner, backbone, args):

        super().__init__()

        self.backbone1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1
        )
        self.shallow_cam = ShallowCAM(args, 256)
        self.backbone2 = nn.Sequential(
            backbone.layer2,
            backbone.layer3,
            # backbone.layer4
        )

    def backbone_modules(self):

        return [self.backbone1, self.backbone2]

    def forward(self, x):

        x = self.backbone1(x)
        intermediate = x = self.shallow_cam(x)
        x = self.backbone2(x)

        return x, intermediate

class ResNetDeepBranch(nn.Module):

    def __init__(self, owner, backbone, args):

        super().__init__()
        self.is_for_test = backbone.is_for_test
        self.backbone = backbone.layer4
        # self.backbone = deepcopy(backbone.layer4)
        self.avgpool = nn.AvgPool2d(kernel_size=(7, 7), stride=1)
        if backbone.is_for_test is False:
            self.fc = nn.Linear(512 * backbone.block.expansion, backbone.class_num)
        

        self.out_dim = 2048

    def backbone_modules(self):

        return [self.backbone]

    def forward(self, x):
        x = self.backbone(x)
        # print('Shape: {}'.format(x.shape))
        # x = self.avgpool(x)
        # print('Shape: {}'.format(x.shape))
        # x = x.view(x.size(0), -1)
        # print('Shape: {}'.format(x.shape))
        if self.is_for_test:
            return x
        # x = self.fc(x)
        # print('Shape: {}'.format(x.shape))
        return x

class ResNetMGNLikeCommonBranch(nn.Module):

    def __init__(self, owner, backbone, args):

        super().__init__()

        self.backbone1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1
        )
        self.shallow_cam = ShallowCAM(args, 256)
        self.backbone2 = nn.Sequential(
            backbone.layer2,
            backbone.layer3[0],
        )

    def backbone_modules(self):

        return [self.backbone1, self.backbone2]

    def forward(self, x):

        x = self.backbone1(x)
        intermediate = x = self.shallow_cam(x)
        x = self.backbone2(x)

        return x, intermediate

class ResNetMGNLikeDeepBranch(nn.Module):

    def __init__(self, owner, backbone, args):

        super().__init__()

        self.backbone = nn.Sequential(
            *deepcopy(backbone.layer3[1:]),
            deepcopy(backbone.layer4)
        )
        self.out_dim = 2048

    def backbone_modules(self):

        return [self.backbone]

    def forward(self, x):
        return self.backbone(x)


class MultiBranchResNet(branches.MultiBranchNetwork):

    def _get_common_branch(self, backbone, args):

        return ResNetCommonBranch(self, backbone, args)

    def _get_middle_subbranch_for(self, backbone, args, last_branch_class):

        return ResNetDeepBranch(self, backbone, args)

class MultiBranchMGNLikeResNet(branches.MultiBranchNetwork):

    def _get_common_branch(self, backbone, args):

        return ResNetMGNLikeCommonBranch(self, backbone, args)

    def _get_middle_subbranch_for(self, backbone, args, last_branch_class):

        return ResNetMGNLikeDeepBranch(self, backbone, args)


def resnet50(num_classes, args, pretrained=True, remove=False, param_path='../../imagenet_models/resnet50-19c8e357.pth', **kwargs):
    """Constructs a ResNet-50 models.
    Args:
        pretrained (bool): If True, returns a models pre-trained on ImageNet
    """
    backbone = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    model = MultiBranchResNet(backbone, args, num_classes)
    if pretrained:
        # backbone.load_state_dict(remove_fc_parameters(load(param_path)), strict=False)
        backbone.load_state_dict(remove_fc_parameters(model_zoo.load_url(model_urls['resnet50'])), strict=False)

        print("using ImageNet pre-trained model to initialize the weight")
    if remove:
        remove_fc(backbone)
    return model


def remove_fc(model):
    del model.fc


def remove_fc_parameters(pretrained_model):

    keys = list(pretrained_model.keys())
    for k in keys:
        if 'fc' in k:
            pretrained_model.pop(k)
    return pretrained_model