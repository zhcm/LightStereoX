# @Time    : 2023/11/27 15:28
# @Author  : zhangchenming
import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.resnet import BasicBlock
from functools import partial

from stereo.modeling.models.lightfast.basic_block_2d import BasicConv2d


class SppBackbone(nn.Module):
    def __init__(self):
        super(SppBackbone, self).__init__()
        self.inplanes = 32
        self.in_conv = nn.Sequential(
            BasicConv2d(in_channels=3, out_channels=16,
                        norm_layer=nn.InstanceNorm2d, act_layer=partial(nn.ReLU, inplace=True),
                        kernel_size=3, padding=1, stride=2),

            BasicConv2d(in_channels=16, out_channels=16,
                        norm_layer=nn.InstanceNorm2d, act_layer=partial(nn.ReLU, inplace=True),
                        kernel_size=3, padding=1),

            BasicConv2d(in_channels=16, out_channels=32,
                        norm_layer=nn.InstanceNorm2d, act_layer=partial(nn.ReLU, inplace=True),
                        kernel_size=3, padding=1)
            )

        self.resblock_1 = self._make_layer(BasicBlock, 64, 3, 2)  # 1/4
        self.resblock_2 = self._make_layer(BasicBlock, 128, 3, 2)  # 1/8

        self.branch1 = nn.Sequential(
            nn.AvgPool2d((16, 16), stride=(16, 16)),
            BasicConv2d(in_channels=128, out_channels=32,
                        norm_layer=nn.InstanceNorm2d, act_layer=partial(nn.ReLU, inplace=True),
                        kernel_size=1)
            )

        self.branch2 = nn.Sequential(
            nn.AvgPool2d((8, 8), stride=(8, 8)),
            BasicConv2d(in_channels=128, out_channels=32,
                        norm_layer=nn.InstanceNorm2d, act_layer=partial(nn.ReLU, inplace=True),
                        kernel_size=1)
            )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d((4, 4), stride=(4, 4)),
            BasicConv2d(in_channels=128, out_channels=32,
                        norm_layer=nn.InstanceNorm2d, act_layer=partial(nn.ReLU, inplace=True),
                        kernel_size=1)
            )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d((2, 2), stride=(2, 2)),
            BasicConv2d(in_channels=128, out_channels=32,
                        norm_layer=nn.InstanceNorm2d, act_layer=partial(nn.ReLU, inplace=True),
                        kernel_size=1)
            )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = BasicConv2d(in_channels=self.inplanes, out_channels=planes * block.expansion,
                                     norm_layer=nn.InstanceNorm2d,
                                     kernel_size=1, stride=stride)

        layers = [block(self.inplanes, planes, stride, downsample, norm_layer=nn.InstanceNorm2d)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=nn.InstanceNorm2d))
        return nn.Sequential(*layers)

    def forward(self, src_stereo):
        _, _, h, w = src_stereo.shape

        output = self.in_conv(src_stereo)  # 1/2
        output_1 = self.resblock_1(output)  # 1/4
        output_2 = self.resblock_2(output_1)  # 1/8

        # spp
        h_spp, w_spp = math.ceil(h / 16), math.ceil(w / 16)
        spp_1 = self.branch1(output_2)
        spp_1 = F.interpolate(spp_1, size=(h_spp, w_spp), mode='bilinear', align_corners=False)
        spp_2 = self.branch2(output_2)
        spp_2 = F.interpolate(spp_2, size=(h_spp, w_spp), mode='bilinear', align_corners=False)
        spp_3 = self.branch3(output_2)
        spp_3 = F.interpolate(spp_3, size=(h_spp, w_spp), mode='bilinear', align_corners=False)
        spp_4 = self.branch4(output_2)
        spp_4 = F.interpolate(spp_4, size=(h_spp, w_spp), mode='bilinear', align_corners=False)
        output_3 = torch.cat([spp_1, spp_2, spp_3, spp_4], dim=1)  # 1/16

        # return [output_3, output_2, output_1, src_stereo]
        return {'scale4': output_3, 'scale3': output_2, 'scale2': output_1}

    @property
    def out_dims(self):
        return {'scale2': 64, 'scale3': 128, 'scale4': 128}
