# @Time    : 2024/3/11 11:29
# @Author  : zhangchenming
import torch
import torch.nn as nn
import torch.nn.functional as F

from stereo.modeling.common.basic_block_2d import BasicConv2d
from stereo.modeling.common.basic_block_3d import BasicConv3d, BasicDeconv3d


class ChannelAtt(nn.Module):
    def __init__(self, cv_chan, im_chan, v):
        super(ChannelAtt, self).__init__()

        self.conv_im = nn.Sequential(
            BasicConv3d(cv_chan, cv_chan, kernel_size=1, norm_layer=nn.BatchNorm3d,act_layer=nn.ReLU),
            BasicConv3d(cv_chan, cv_chan, kernel_size=1, norm_layer=nn.BatchNorm3d),
        )

        self.avgpool = nn.AdaptiveAvgPool3d((v, 1, 1))
        self.conv_cv = nn.Sequential(
            BasicConv3d(cv_chan, cv_chan, kernel_size=1, norm_layer=nn.BatchNorm3d,act_layer=nn.ReLU),
            BasicConv3d(cv_chan, cv_chan, kernel_size=1, norm_layer=nn.BatchNorm3d),
        )

        # self.conv1 = nn.Conv2d(im_chan, 1, 1)
        # self.softmax = nn.Softmax(dim=2)
        #
        # self.channel_conv = nn.Sequential(
        #     nn.Conv3d(in_channels=cv_chan, out_channels=cv_chan, kernel_size=1),
        #     nn.LayerNorm([cv_chan, v, 1, 1]),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(in_channels=cv_chan, out_channels=cv_chan, kernel_size=1)
        # )

    def forward(self, cost, im):
        im_feat = self.conv_im(im)
        cost_feat = self.avgpool(cost)
        cost_feat = self.conv_cv(cost_feat)
        channel_term = cost_feat + im_feat
        cost = cost * torch.sigmoid(channel_term)
        return cost

        bz, c, h, w = im.shape
        context_mask = self.conv1(im)  # [bz, 1, h, w]
        context_mask = context_mask.reshape(bz, 1, h * w)  # [bz, 1, hw]
        context_mask = self.softmax(context_mask)  # [bz, 1, hw]
        context_mask = context_mask.unsqueeze(-1)  # [bz, 1, hw, 1]

        bz, c, v, h, w = cost.shape
        cost_x = cost.reshape(bz, c, v, h * w)
        cost_x = cost_x.transpose(1, 2)  # [bz, v, c, hw]

        context = torch.matmul(cost_x, context_mask)  # [bz, v, c, 1]
        context = context.transpose(1, 2).unsqueeze(-1)  # [bz, c, v, 1, 1]
        channel_term = self.channel_conv(context)

        # cost = cost * torch.sigmoid(channel_term)
        cost = cost + channel_term
        return cost


class Aggregation(nn.Module):
    def __init__(self, in_channels=1, gce=True, backbone_channels=None):
        super(Aggregation, self).__init__()

        self.gce = gce
        if backbone_channels is None:
            backbone_channels = [96, 64, 192, 160]

        channels = [8, 16, 32, 40]
        # x4
        self.conv_stem = BasicConv3d(in_channels, channels[0], kernel_size=3, stride=1, padding=1,
                                     norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU)

        self.conv_down = nn.ModuleList()
        self.conv_up = nn.ModuleList()
        self.conv_agg = nn.ModuleList()
        for i in range(3):
            self.conv_down.append(
                nn.Sequential(
                    BasicConv3d(channels[i], channels[i + 1], kernel_size=3, stride=2, padding=1,
                                norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU),
                    BasicConv3d(channels[i + 1], channels[i + 1], kernel_size=3, stride=1, padding=1,
                                norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU)))

            # ch_out = 1 if i == 2 else channels[-2 - i]
            norm_layer = None if i == 2 else nn.BatchNorm3d
            act_layer = None if i == 2 else nn.LeakyReLU
            self.conv_up.append(
                BasicDeconv3d(channels[-1 - i], channels[-2 - i], kernel_size=4, stride=2, padding=1,
                              norm_layer=norm_layer, act_layer=act_layer))

            self.conv_agg.append(
                nn.Sequential(
                    BasicConv3d(channels[-2 - i] * 2, channels[-2 - i], kernel_size=1,
                                norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU),
                    BasicConv3d(channels[-2 - i], channels[-2 - i], kernel_size=3, padding=1,
                                norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU),
                    BasicConv3d(channels[-2 - i], channels[-2 - i], kernel_size=3, padding=1,
                                norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU)))

        if gce:
            self.channel_att_stem = ChannelAtt(channels[0], backbone_channels[0], 48)
            self.channel_att_down = nn.ModuleList()
            self.channel_att_up = nn.ModuleList()
            for i in range(3):
                self.channel_att_down.append(
                    ChannelAtt(channels[i + 1], backbone_channels[i + 1], int(24/2**i)))

                self.channel_att_up.append(
                    ChannelAtt(channels[-2 - i], backbone_channels[-2 - i], int(12*2**i)))

        self.conv_agg.pop(-1)
        if gce:
            self.channel_att_up.pop(-1)

        self.last_conv = nn.Conv3d(channels[0], 1, 3, 1, 1, bias=False)

    def forward(self, features_left, cost):
        cost = self.conv_stem(cost)
        # if self.gce:
        #     cost = self.channel_att_stem(cost, features_left[0])  # [bz, 8, max_disp/4, H/4, W/4]
        cost_feature = [cost]

        for i in range(3):
            cost = self.conv_down[i](cost)
            if self.gce:
                cost = self.channel_att_down[i](cost, features_left[i])
            cost_feature.append(cost)

        res_cost = cost_feature[-1]
        for i in range(3):
            res_cost = self.conv_up[i](res_cost)
            if res_cost.shape[-3:] != cost_feature[-2 - i].shape[-3:]:
                res_cost = F.interpolate(res_cost, size=cost_feature[-2 - i].shape[-3:], mode='nearest')
            if i == 2: break
            res_cost = torch.cat([res_cost, cost_feature[-2 - i]], 1)
            res_cost = self.conv_agg[i](res_cost)
            if self.gce:
                res_cost = self.channel_att_up[i](res_cost, features_left[-2 - i])

        res_cost = self.last_conv(res_cost)
        return res_cost
