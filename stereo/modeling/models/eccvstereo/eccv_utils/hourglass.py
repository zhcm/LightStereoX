import torch
import torch.nn as nn
import torch.nn.functional as F
from stereo.modeling.common.basic_block_3d import BasicConv3d, BasicDeconv3d


class Hourglass(nn.Module):
    def __init__(self, in_channels):
        super(Hourglass, self).__init__()

        self.conv1 = nn.Sequential(
            BasicConv3d(in_channels, in_channels * 2,
                        norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU,
                        kernel_size=3, padding=1, stride=2, dilation=1),
            BasicConv3d(in_channels * 2, in_channels * 2,
                        norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU,
                        kernel_size=3, padding=1, stride=1, dilation=1),
        )

        self.conv2 = nn.Sequential(
            BasicConv3d(in_channels * 2, in_channels * 4,
                        norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU,
                        kernel_size=3, padding=1, stride=2, dilation=1),
            BasicConv3d(in_channels * 4, in_channels * 4,
                        norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU,
                        kernel_size=3, padding=1, stride=1, dilation=1))

        self.conv3 = nn.Sequential(
            BasicConv3d(in_channels * 4, in_channels * 6,
                        norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU,
                        kernel_size=3, padding=1, stride=2, dilation=1),
            BasicConv3d(in_channels * 6, in_channels * 6,
                        norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU,
                        kernel_size=3, padding=1, stride=1, dilation=1))

        self.conv3_up = BasicDeconv3d(in_channels * 6, in_channels * 4,
                                      norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU,
                                      kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicDeconv3d(in_channels * 4, in_channels * 2,
                                      norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU,
                                      kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicDeconv3d(in_channels * 2, in_channels,
                                      norm_layer=None, act_layer=None,
                                      kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(
            BasicConv3d(in_channels * 8, in_channels * 4,
                        norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU,
                        kernel_size=1, padding=0, stride=1),
            BasicConv3d(in_channels * 4, in_channels * 4,
                        norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU,
                        kernel_size=3, padding=1, stride=1),
            BasicConv3d(in_channels * 4, in_channels * 4,
                        norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU,
                        kernel_size=3, padding=1, stride=1), )

        self.agg_1 = nn.Sequential(
            BasicConv3d(in_channels * 4, in_channels * 2,
                        norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU,
                        kernel_size=1, padding=0, stride=1),
            BasicConv3d(in_channels * 2, in_channels * 2,
                        norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU,
                        kernel_size=3, padding=1, stride=1),
            BasicConv3d(in_channels * 2, in_channels * 2,
                        norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU,
                        kernel_size=3, padding=1, stride=1))

    def forward(self, x, return_multi=False):  # [bz, c, disp/4, H/4, W/4]
        conv1 = self.conv1(x)  # [bz, 2c, disp/8, H/8, W/8]
        conv2 = self.conv2(conv1)  # [bz, 4c, disp/16, H/16, W/16]
        conv3 = self.conv3(conv2)  # [bz, 6c, disp/32, H/32, W/32]

        conv3_up = self.conv3_up(conv3)  # [bz, 4c, disp/16, H/16, W/16]
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)  # [bz, 4c, disp/16, H/16, W/16]

        conv2_up = self.conv2_up(conv2)  # [bz, 2c, disp/8, H/8, W/8]
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)

        conv = self.conv1_up(conv1)  # [bz, c, disp/4, H/4, W/4]

        if return_multi:
            return [conv, conv1, conv2]
        else:
            return conv


class Hourglass2D(nn.Module):
    def __init__(self, in_channels):
        super(Hourglass2D, self).__init__()

        self.expanse_ratio = 2

        self.conv1 = MobileV2Residual(in_channels, in_channels * 2, stride=2, expanse_ratio=self.expanse_ratio)

        self.conv2 = MobileV2Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)

        self.conv3 = MobileV2Residual(in_channels * 2, in_channels * 4, stride=2, expanse_ratio=self.expanse_ratio)

        self.conv4 = MobileV2Residual(in_channels * 4, in_channels * 4, stride=1, expanse_ratio=self.expanse_ratio)

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels))

        self.redir1 = MobileV2Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
        self.redir2 = MobileV2Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6


class MobileV2Residual(nn.Module):
    def __init__(self, inp, oup, stride, expanse_ratio, dilation=1):
        super(MobileV2Residual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expanse_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        pad = dilation

        if expanse_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, pad, dilation=dilation, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, pad, dilation=dilation, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)