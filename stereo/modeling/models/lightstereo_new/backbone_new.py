# @Time    : 2024/3/10 10:21
# @Author  : zhangchenming
import timm
import torch
import torch.nn as nn

from functools import partial
from torch.hub import load_state_dict_from_url
from stereo.modeling.common.basic_block_2d import BasicConv2d, BasicDeconv2d
from .repvit import repvit_m0_9, repvit_m0_6


class FPNLayer(nn.Module):
    def __init__(self, chan_low, chan_high):
        super().__init__()
        self.deconv = BasicDeconv2d(chan_low, chan_high, kernel_size=4, stride=2, padding=1,
                                     norm_layer=nn.BatchNorm2d,
                                     act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True))

        self.conv = BasicConv2d(chan_high * 2, chan_high * 2, kernel_size=3, padding=1,
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

        # model = timm.create_model("mobilenetv3_large_100", pretrained=True, features_only=True)
        # self.conv_stem = model.conv_stem
        # self.bn1 = model.bn1
        # self.act1 = model.act1
        # self.block0 = model.blocks[0]
        # self.block1 = model.blocks[1]
        # self.block2 = model.blocks[2]
        # self.block3 = model.blocks[3:5]
        # self.block4 = model.blocks[5]
        # channels = [160, 112, 40, 24, 16]

        # model = timm.create_model('mobilenetv2_100', pretrained=True, features_only=True)
        # self.conv_stem = model.conv_stem
        # self.bn1 = model.bn1
        # self.act1 = model.act1
        # self.block0 = model.blocks[0]
        # self.block1 = model.blocks[1]
        # self.block2 = model.blocks[2]
        # self.block3 = model.blocks[3:5]
        # self.block4 = model.blocks[5]
        # channels = [160, 96, 32, 24]

        self.fpn_layer4 = FPNLayer(channels[0], channels[1])
        self.fpn_layer3 = FPNLayer(channels[1] * 2, channels[2])
        self.fpn_layer2 = FPNLayer(channels[2] * 2, channels[3])

        self.out_conv = BasicConv2d(channels[3] * 2, channels[3] * 2,
                                    kernel_size=3, padding=1, padding_mode="replicate",
                                    norm_layer=nn.InstanceNorm2d)

        self.branch_1 = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=2, padding=1,
                        norm_layer=nn.InstanceNorm2d, act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.InstanceNorm2d, act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)))
        self.branch_2 = nn.Sequential(
            BasicConv2d(32, 48, kernel_size=3, stride=2, padding=1,
                        norm_layer=nn.InstanceNorm2d, act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)),
            BasicConv2d(48, 48, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.InstanceNorm2d, act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)))

    def forward(self, images):
        feat = images
        for i in range(0, self.block_index[0]):
            feat = self.features[i](feat)
        c2 = feat  # [bz, 48, H/4, W/4]
        for i in range(self.block_index[0], self.block_index[1]):
            feat = self.features[i](feat)
        c3 = feat  # [bz, 96, H/8, W/8]
        for i in range(self.block_index[1], self.block_index[2]):
            feat = self.features[i](feat)
        c4 = feat  # [bz, 192, H/16, W/16]
        for i in range(self.block_index[2], self.block_index[3]):
            feat = self.features[i](feat)
        c5 = feat  # [bz, 384, H/32, W/32]

        # c1 = self.act1(self.bn1(self.conv_stem(images)))  # [bz, 32, H/2, W/2]
        # c1 = self.block0(c1)  # [bz, 16, H/2, W/2]
        # c2 = self.block1(c1)  # [bz, 24, H/4, W/4]
        # c3 = self.block2(c2)  # [bz, 32, H/8, W/8]
        # c4 = self.block3(c3)  # [bz, 96, H/16, W/16]
        # c5 = self.block4(c4)  # [bz, 160, H/32, W/32]

        p4 = self.fpn_layer4(c5, c4)  # [bz, 96, H/16, W/16]
        p3 = self.fpn_layer3(p4, c3)  # [bz, 32, H/8, W/8]
        p2 = self.fpn_layer2(p3, c2)  # [bz, 24, H/4, W/4]
        p2 = self.out_conv(p2)

        b1 = self.branch_1(images)
        b2 = self.branch_2(b1)

        p2 = torch.cat([p2, b2], 1)

        return [b1, p2, p3, p4, c5]
