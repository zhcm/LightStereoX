# @Time    : 2024/3/10 10:21
# @Author  : zhangchenming
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from stereo.modeling.common.basic_block_2d import BasicConv2d, BasicDeconv2d


class FPNLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        channels = 128
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1)
        self.conv2 = BasicConv2d(channels, channels, kernel_size=3, padding=1,
                                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU)

    def forward(self, x, y):
        x = self.conv1(x)
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=True)
        feat = x + y
        feat = self.conv2(feat)
        return feat


class PANLayer(nn.Module):
    def __init__(self):
        super().__init__()
        channels = 128
        self.conv1 = BasicConv2d(channels, channels, kernel_size=3, stride=2, padding=1,
                                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU)
        self.conv2 = BasicConv2d(channels, channels, kernel_size=3, padding=1,
                                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU)

    def forward(self, x, y):
        x = self.conv1(x)
        feat = x + y
        feat = self.conv2(feat)
        return feat


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        model = timm.create_model('mobilenetv2_100', pretrained=True, features_only=True)
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = model.act1
        self.block0 = model.blocks[0]
        self.block1 = model.blocks[1]
        self.block2 = model.blocks[2]
        self.block3 = model.blocks[3:5]
        self.block4 = model.blocks[5]

        self.top_layer = nn.Conv2d(160, 128, kernel_size=1)
        self.fpn_layer1 = FPNLayer(96)
        self.fpn_layer2 = FPNLayer(32)
        self.fpn_layer3 = FPNLayer(24)

        self.pan_layer1 = PANLayer()
        self.pan_layer2 = PANLayer()
        self.pan_layer3 = PANLayer()

    def forward(self, images):
        c1 = self.act1(self.bn1(self.conv_stem(images)))  # [bz, 32, H/2, W/2]
        c1 = self.block0(c1)  # [bz, 16, H/2, W/2]
        c2 = self.block1(c1)  # [bz, 24, H/4, W/4]
        c3 = self.block2(c2)  # [bz, 32, H/8, W/8]
        c4 = self.block3(c3)  # [bz, 96, H/16, W/16]
        c5 = self.block4(c4)  # [bz, 160, H/32, W/32]

        p5 = self.top_layer(c5)
        p4 = self.fpn_layer1(c4, p5)
        p3 = self.fpn_layer2(c3, p4)
        p2 = self.fpn_layer3(c2, p3)

        n2 = p2
        n3 = self.pan_layer1(n2, p3)
        n4 = self.pan_layer2(n3, p4)
        n5 = self.pan_layer3(n4, p5)

        return [n2, n3, n4, n5], c1
