# @Time    : 2024/3/10 10:21
# @Author  : zhangchenming
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from stereo.modeling.common.basic_block_2d import BasicConv2d, BasicDeconv2d
from .repvit import repvit_m0_9, repvit_m0_6


class DeConv2x(nn.Module):
    def __init__(self, channels_x, channels_y):
        super().__init__()
        self.conv1 = BasicDeconv2d(channels_x, channels_y,
                                   kernel_size=4, stride=2, padding=1,
                                   norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU)
        self.conv2 = BasicConv2d(channels_y * 2, channels_y * 2,
                                 kernel_size=3, stride=1, padding=1,
                                 norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU)

    def forward(self, x, y):
        x = self.conv1(x)
        if x.shape != y.shape: x = F.interpolate(x, size=(y.shape[-2], y.shape[-1]), mode='nearest')
        x = torch.cat([x, y], 1)
        x = self.conv2(x)
        return x


class FuseFeat(nn.Module):
    def __init__(self):
        super().__init__()

        chans = [40, 80, 160, 320]
        self.deconv1 = DeConv2x(chans[3], chans[2])
        self.deconv2 = DeConv2x(chans[2] * 2, chans[1])
        self.deconv3 = DeConv2x(chans[1] * 2, chans[0])

        self.conv4 = BasicConv2d(chans[0] * 2, chans[0] * 2, kernel_size=3, stride=1, padding=1,
                                 norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU)

    def forward(self, feat):
        x4, x8, x16, x32 = feat
        x16 = self.deconv1(x32, x16)  # [bz, 192, H/16, W/16]
        x8 = self.deconv2(x16, x8)    # [bz, 64, H/8, W/8]
        x4 = self.deconv3(x8, x4)    # [bz, 48, H/4, W/4]
        x4 = self.conv4(x4)  # [bz, 48, H/4, W/4]
        return [x4, x8, x16, x32]


class Feature(nn.Module):
    def __init__(self):
        super().__init__()
        model = repvit_m0_6()
        checkpoint = load_state_dict_from_url('repvit_m0_6_distill_300e.pth', map_location='cpu')
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict, strict=False)
        self.features = model.features

        self.block_index = [3, 6, 16, 18]
        # [4, 8, 24, 27]
        # [3, 6, 16, 18]

    def forward(self, x):
        feat = x
        for i in range(0, self.block_index[0]):
            feat = self.features[i](feat)
        x4 = feat  # [bz, 48, H/4, W/4]

        for i in range(self.block_index[0], self.block_index[1]):
            feat = self.features[i](feat)
        x8 = feat  # [bz, 96, H/8, W/8]

        for i in range(self.block_index[1], self.block_index[2]):
            feat = self.features[i](feat)
        x16 = feat  # [bz, 192, H/16, W/16]

        for i in range(self.block_index[2], self.block_index[3]):
            feat = self.features[i](feat)
        x32 = feat  # [bz, 384, H/32, W/32]

        return [x4, x8, x16, x32]


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat = Feature()
        self.fuse = FuseFeat()

        self.stem_2 = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=2, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU))
        self.stem_4 = nn.Sequential(
            BasicConv2d(32, 48, kernel_size=3, stride=2, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(48, 48, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU))

    def forward(self, images):
        features = self.feat(images)
        features = self.fuse(features)

        stem_2x = self.stem_2(images)  # [bz, 32, H/2, W/2]
        stem_4x = self.stem_4(stem_2x)  # [bz, 48, H/4, W/4]
        features[0] = torch.cat([features[0], stem_4x], dim=1)  # [bz, 96, H/4, W/4]

        return features, stem_2x
