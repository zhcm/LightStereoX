# @Time    : 2024/3/11 11:29
# @Author  : zhangchenming
import torch
import torch.nn as nn

from functools import partial
from stereo.modeling.common.basic_block_2d import BasicConv2d
from stereo.modeling.common.basic_block_3d import BasicConv3d, BasicDeconv3d


class DWRLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        mid_channels = in_channels // 2

        self.first_bottleneck = nn.Sequential(
            BasicConv3d(mid_channels, mid_channels, kernel_size=[1, 3, 1], padding=[0, 1, 0],
                        act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)),
            BasicConv3d(mid_channels, mid_channels, kernel_size=[1, 1, 3], padding=[0, 0, 1],
                        norm_layer=nn.BatchNorm3d,
                        act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)),

            BasicConv3d(mid_channels, mid_channels, kernel_size=[1, 3, 1], dilation=[1, 2, 1], padding=[0, 2, 0],
                        act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)),
            BasicConv3d(mid_channels, mid_channels, kernel_size=[1, 1, 3], dilation=[1, 1, 2], padding=[0, 0, 2],
                        norm_layer=nn.BatchNorm3d,
                        act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True))
        )

        self.second_bottleneck = nn.Sequential(
            BasicConv3d(mid_channels, mid_channels, kernel_size=[1, 3, 1], padding=[0, 1, 0],
                        act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)),
            BasicConv3d(mid_channels, mid_channels, kernel_size=[1, 1, 3], padding=[0, 0, 1],
                        norm_layer=nn.BatchNorm3d,
                        act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)),

            BasicConv3d(mid_channels, mid_channels, kernel_size=[1, 3, 1], dilation=[1, 2, 1], padding=[0, 2, 0],
                        act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)),
            BasicConv3d(mid_channels, mid_channels, kernel_size=[1, 1, 3], dilation=[1, 1, 2], padding=[0, 0, 2],
                        norm_layer=nn.BatchNorm3d,
                        act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True))
        )

    def forward(self, x):
        splits = torch.chunk(x, 2, dim=1)
        return feat


class ChannelAtt(nn.Module):
    def __init__(self, cv_chan, im_chan, v):
        super(ChannelAtt, self).__init__()

        self.im_att = nn.Sequential(
            BasicConv2d(im_chan, im_chan // 2, kernel_size=1, stride=1, padding=0,
                        norm_layer=nn.BatchNorm2d,
                        act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)),
            nn.Conv2d(im_chan // 2, cv_chan, 1))

    def forward(self, cost, im):
        channel_att = self.im_att(im).unsqueeze(2)
        cost = torch.sigmoid(channel_att) * cost
        return cost


class FPNLayer(nn.Module):
    def __init__(self, chan_low, chan_high):
        super().__init__()
        self.deconv = BasicDeconv3d(chan_low, chan_high, kernel_size=4, stride=2, padding=1,
                                     norm_layer=nn.BatchNorm3d,
                                     act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True))

        self.conv2 = BasicConv3d(chan_high * 2, chan_high, kernel_size=1,
                                 norm_layer=nn.BatchNorm3d,
                                 act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True))

    def forward(self, low, high):
        low = self.deconv(low)
        feat = torch.cat([low, high], 1)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv4(feat)
        return feat


class Aggregation(nn.Module):
    def __init__(self, in_channels=1, backbone_channels=None):
        super(Aggregation, self).__init__()

        if backbone_channels is None:
            backbone_channels = [24*4, 32*2, 96*2, 160]

        channels = [8, 16, 32, 48]
        self.conv1 = BasicConv3d(in_channels, channels[0], kernel_size=3, stride=1, padding=1,
                                 norm_layer=nn.BatchNorm3d,
                                 act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True))
        self.attention1 = ChannelAtt(channels[0], backbone_channels[0], 48)

        self.conv2 = nn.Sequential(
            BasicConv3d(channels[0], channels[1], kernel_size=3, stride=2, padding=1,
                        norm_layer=nn.BatchNorm3d,
                        act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)),
            BasicConv3d(channels[1], channels[1], kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.BatchNorm3d,
                        act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)))
        self.attention2 = ChannelAtt(channels[1], backbone_channels[1], 24)

        self.conv3 = nn.Sequential(
            BasicConv3d(channels[1], channels[2], kernel_size=3, stride=2, padding=1,
                        norm_layer=nn.BatchNorm3d,
                        act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)),
            BasicConv3d(channels[2], channels[2], kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.BatchNorm3d,
                        act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)))
        self.attention3 = ChannelAtt(channels[2], backbone_channels[2], 12)

        self.conv4 = nn.Sequential(
            BasicConv3d(channels[2], channels[3], kernel_size=3, stride=2, padding=1,
                        norm_layer=nn.BatchNorm3d,
                        act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)),
            BasicConv3d(channels[3], channels[3], kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.BatchNorm3d,
                        act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)))
        self.attention4 = ChannelAtt(channels[3], backbone_channels[3], 6)

        self.fpn1 = FPNLayer(channels[3], channels[2])
        self.fpn_attention1 = ChannelAtt(channels[2], backbone_channels[2], 12)

        self.fpn2 = FPNLayer(channels[2], channels[1])
        self.fpn_attention2 = ChannelAtt(channels[1], backbone_channels[1], 24)

        self.last_deconv = BasicDeconv3d(channels[1], channels[0], kernel_size=4, padding=1, stride=2)
        self.last_conv = BasicConv3d(channels[0], 1, kernel_size=3, padding=1)

    def forward(self, features_left, cost):
        c2 = self.conv1(cost)  # [bz, 8, max_disp/4, H/4, W/4]
        c2 = self.attention1(c2, features_left[1])

        c3 = self.conv2(c2)  # [bz, 16, max_disp/8, H/8, W/8]
        c3 = self.attention2(c3, features_left[2])

        c4 = self.conv3(c3)  # [bz, 32, max_disp/16, H/16, W/16]
        c4 = self.attention3(c4, features_left[3])

        c5 = self.conv4(c4)  # [bz, 48, max_disp/32, H/32, W/32]
        c5 = self.attention4(c5, features_left[4])

        p4 = self.fpn1(c5, c4)  # [bz, 32, max_disp/16, H/16, W/16]
        p4 = self.fpn_attention1(p4, features_left[3])

        p3 = self.fpn2(p4, c3)  # [bz, 16, max_disp/8, H/8, W/8]
        p3 = self.fpn_attention2(p3, features_left[2])

        p2 = self.last_deconv(p3)
        result = self.last_conv(p2)
        return result
