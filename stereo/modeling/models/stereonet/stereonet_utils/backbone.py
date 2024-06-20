# @Time    : 2024/1/5 17:02
# @Author  : zhangchenming
import torch.nn as nn

from functools import partial
from stereo.modeling.common.basic_block_2d import BasicConv2d


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        self.stride = stride

        norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers= []
        if expand_ratio != 1:
            # pw
            layers.append(BasicConv2d(in_channels=inp, out_channels=hidden_dim,
                                      norm_layer=norm_layer, act_layer=nn.ReLU6,
                                      kernel_size=1)
            )
        layers.extend(
            [
                # dw
                BasicConv2d(in_channels=hidden_dim, out_channels=hidden_dim,
                            norm_layer=norm_layer, act_layer=nn.ReLU6,
                            kernel_size=3, padding=1, stride=stride, groups=hidden_dim),
                # pw-linear
                BasicConv2d(in_channels=hidden_dim, out_channels=oup,
                            norm_layer=norm_layer, act_layer=None,
                            kernel_size=1),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, padding=0, dilation=1):
        super().__init__()
        self.block_1 = BasicConv2d(in_channels=channels, out_channels=channels,
                                   norm_layer=nn.BatchNorm2d, act_layer=partial(nn.LeakyReLU, negative_slope=0.2),
                                   kernel_size=kernel_size, padding=padding, dilation=dilation)

        self.block_2 = BasicConv2d(in_channels=channels, out_channels=channels,
                                   norm_layer=nn.BatchNorm2d, act_layer=None,
                                   kernel_size=kernel_size, padding=padding, dilation=dilation)

        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        res = self.block_1(x)
        res = self.block_2(res)
        out = res + x
        out = self.activation(out)
        return out


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, k_downsamp_layers):
        super().__init__()
        net = []
        for _ in range(k_downsamp_layers):
            net.append(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=2, padding=2))
            in_channels = out_channels
        for _ in range(6):
            net.append(ResBlock(channels=out_channels, kernel_size=3, padding=1))

        net.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        x = self.net(x)
        return x
