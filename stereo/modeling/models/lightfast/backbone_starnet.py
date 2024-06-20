# @Time    : 2024/3/10 10:21
# @Author  : zhangchenming
import timm
import torch
import torch.nn as nn
from functools import partial

from stereo.modeling.common.basic_block_2d import BasicConv2d, BasicDeconv2d
from .starnet import starnet_s1, starnet_s2, starnet_s3, starnet_s4

class FPNLayer(nn.Module):
    def __init__(self, chan_low, chan_high):
        super().__init__()
        self.deconv = BasicDeconv2d(chan_low, chan_high, kernel_size=4, stride=2, padding=1,
                                     norm_layer=nn.BatchNorm2d,
                                     act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True))

        self.conv = BasicConv2d(chan_high * 2, chan_high, kernel_size=3, padding=1,
                                 norm_layer=nn.BatchNorm2d,
                                act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True))

    def forward(self, low, high):
        low = self.deconv(low)
        feat = torch.cat([high, low], 1)
        feat = self.conv(feat)
        return feat

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        model = starnet_s1(pretrained=True)
        self.stem = model.stem
        self.stage0 = model.stages[0]
        self.stage1 = model.stages[1]
        self.stage2 = model.stages[2]
        self.stage3 = model.stages[3]

        channels = [192, 96, 48, 24]
        # channels = [256, 128, 64, 32]

        self.fpn_layer4 = FPNLayer(channels[0], channels[1])
        self.fpn_layer3 = FPNLayer(channels[1], channels[2])
        self.fpn_layer2 = FPNLayer(channels[2], channels[3])

        self.out_conv = BasicConv2d(channels[3], channels[3],
                                    kernel_size=3, padding=1, padding_mode="replicate",
                                    norm_layer=nn.InstanceNorm2d)

    def forward(self, images):
        c1 = self.stem(images)
        c2 = self.stage0(c1)
        c3 = self.stage1(c2)
        c4 = self.stage2(c3)
        c5 = self.stage3(c4)

        p4 = self.fpn_layer4(c5, c4)
        p3 = self.fpn_layer3(p4, c3)
        p2 = self.fpn_layer2(p3, c2)
        p2 = self.out_conv(p2)

        return [p2, p3, p4, c5]
