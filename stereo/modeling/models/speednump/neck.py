# @Time    : 2024/3/10 10:21
# @Author  : zhangchenming
import timm
import torch
import torch.nn as nn

from functools import partial
from .basic_block_2d import BasicConv2d, BasicDeconv2d


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


class Neck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fpn_layer4 = FPNLayer(channels[3], channels[2])
        self.fpn_layer3 = FPNLayer(channels[2], channels[1])
        self.fpn_layer2 = FPNLayer(channels[1], channels[0])

        self.out_conv = BasicConv2d(channels[0], channels[0],
                                    kernel_size=3, padding=1, padding_mode="replicate",
                                    norm_layer=nn.InstanceNorm2d)

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, features):
        c2, c3, c4, c5 = features
        p4 = self.fpn_layer4(c5.float(), c4.float())  # [bz, 96, H/16, W/16]
        p3 = self.fpn_layer3(p4, c3.float())  # [bz, 32, H/8, W/8]
        p2 = self.fpn_layer2(p3, c2.float())  # [bz, 24, H/4, W/4]
        p2 = self.out_conv(p2)  # [bz, 24, H/4, W/4]

        return [p2, p3, p4, c5]
