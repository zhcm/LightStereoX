# @Time    : 2024/3/10 10:21
# @Author  : zhangchenming
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from functools import partial

from stereo.modeling.common.basic_block_2d import BasicConv2d, BasicDeconv2d
from ..lightstereo.repvit import repvit_m0_9, repvit_m0_6


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
        model = repvit_m0_6()
        checkpoint = load_state_dict_from_url('repvit_m0_6_distill_300e.pth', map_location='cpu')
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict, strict=False)
        self.features = model.features
        self.block_index = [3, 6, 16, 18]

        channels = [320, 160, 80, 40]

        self.fpn_layer4 = FPNLayer(channels[0], channels[1])
        self.fpn_layer3 = FPNLayer(channels[1], channels[2])
        self.fpn_layer2 = FPNLayer(channels[2], channels[3])

        self.out_conv = BasicConv2d(channels[3], channels[3],
                                    kernel_size=3, padding=1, padding_mode="replicate",
                                    norm_layer=nn.InstanceNorm2d)

    def forward(self, images):
        feat = images
        for i in range(0, self.block_index[0]):
            feat = self.features[i](feat)
        c2 = feat

        for i in range(self.block_index[0], self.block_index[1]):
            feat = self.features[i](feat)
        c3 = feat

        for i in range(self.block_index[1], self.block_index[2]):
            feat = self.features[i](feat)
        c4 = feat

        for i in range(self.block_index[2], self.block_index[3]):
            feat = self.features[i](feat)
        c5 = feat

        p4 = self.fpn_layer4(c5, c4)
        p3 = self.fpn_layer3(p4, c3)
        p2 = self.fpn_layer2(p3, c2)
        p2 = self.out_conv(p2)

        return [p2, p3, p4, c5]
