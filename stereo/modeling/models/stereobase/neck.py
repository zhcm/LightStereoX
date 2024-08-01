# @Time    : 2024/3/10 10:21
# @Author  : zhangchenming
import torch
import torch.nn as nn
from stereo.modeling.models.lightstereo.basic_block_2d import BasicConv2d, BasicDeconv2d


class FPNLayer(nn.Module):
    def __init__(self, chan_low, chan_high, norm_layer):
        super().__init__()
        self.conv1 = BasicDeconv2d(chan_low, chan_high,
                                   norm_layer=norm_layer, act_layer=nn.LeakyReLU,
                                   kernel_size=4, stride=2, padding=1)

        self.conv2 = BasicConv2d(chan_high * 2, chan_high * 2,
                                 norm_layer=norm_layer, act_layer=nn.LeakyReLU,
                                 kernel_size=3, stride=1, padding=1)

    def forward(self, low, high):
        low = self.conv1(low)
        feat = torch.cat([low, high], 1)
        feat = self.conv2(feat)
        return feat


class Neck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fpn_layer4 = FPNLayer(channels[3], channels[2], nn.InstanceNorm2d)
        self.fpn_layer3 = FPNLayer(channels[2] * 2, channels[1], nn.InstanceNorm2d)
        self.fpn_layer2 = FPNLayer(channels[1] * 2, channels[0], nn.InstanceNorm2d)

        self.out_conv = nn.Sequential(
            nn.Conv2d(channels[0] * 2, channels[0] * 2, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels[0] * 2),
            nn.LeakyReLU()
        )

    def forward(self, features):
        c2, c3, c4, c5 = features
        p4 = self.fpn_layer4(c5, c4)  # [bz, 96, H/16, W/16]
        p3 = self.fpn_layer3(p4, c3)  # [bz, 32, H/8, W/8]
        p2 = self.fpn_layer2(p3, c2)  # [bz, 24, H/4, W/4]
        p2 = self.out_conv(p2)  # [bz, 24, H/4, W/4]

        return [p2, p3, p4, c5]
