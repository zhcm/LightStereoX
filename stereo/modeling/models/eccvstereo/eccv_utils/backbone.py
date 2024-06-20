# @Time    : 2023/8/29 03:01
# @Author  : zhangchenming
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from stereo.modeling.common.basic_block_2d import BasicConv2d, BasicDeconv2d


class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        pretrained = True
        model = timm.create_model('mobilenetv2_100', pretrained=pretrained, features_only=True)

        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = model.act1

        self.block0 = model.blocks[0]
        self.block1 = model.blocks[1]
        self.block2 = model.blocks[2]
        self.block3 = model.blocks[3:5]
        self.block4 = model.blocks[5]

        self.conv1 = BasicConv2d(160 + 96, 96, kernel_size=3, stride=1, padding=1,
                                     norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU)
        self.conv2 = BasicConv2d(96 + 32, 32, kernel_size=3, stride=1, padding=1,
                                    norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU)
        self.conv3 = BasicConv2d(32 + 24, 48, kernel_size=3, stride=1, padding=1,
                                   norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU)

        self.conv_last = BasicConv2d(48, 48, kernel_size=3, stride=1, padding=1,
                                     norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU)

    def forward(self, x):
        x2 = self.act1(self.bn1(self.conv_stem(x)))
        x2 = self.block0(x2)
        x4 = self.block1(x2)
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)

        x32_16 = F.interpolate(x32, size=x16.shape[2:], mode='bilinear', align_corners=False)
        x16 = self.conv1(torch.cat([x32_16, x16], dim=1))

        x16_8 = F.interpolate(x16, size=x8.shape[2:], mode='bilinear', align_corners=False)
        x8 = self.conv2(torch.cat([x16_8, x8], dim=1))

        x8_4 = F.interpolate(x8, size=x4.shape[2:], mode='bilinear', align_corners=False)
        x4 = self.conv3(torch.cat([x8_4, x4], dim=1))

        x4 = self.conv_last(x4)
        return x4
