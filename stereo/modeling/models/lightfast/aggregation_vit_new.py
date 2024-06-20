# @Time    : 2024/3/11 11:29
# @Author  : zhangchenming
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from stereo.modeling.common.basic_block_2d import BasicConv2d
from .efficientvit import EfficientViTBlock, Residual, Conv2d_BN, FFN, PatchMerging


class Aggregation(nn.Module):
    def __init__(self, in_channels, left_att):
        super(Aggregation, self).__init__()

        self.expanse_ratio = 4

        attn_ratio = [1.0, 2.0, 4.0]
        embed_dim = [48, 96, 192]
        resolution = 32
        conv0 = [EfficientViTBlock('s', embed_dim[0], 16, 3, attn_ratio[0], resolution, 7)
                 for i in range(3)]
        self.conv0 = nn.Sequential(*conv0)

        self.conv1 = nn.Sequential(
            Residual(Conv2d_BN(embed_dim[0], embed_dim[0], 3, 1, 1, groups=embed_dim[0], resolution=resolution)),
            Residual(FFN(embed_dim[0], int(embed_dim[0] * 2), resolution)),
            PatchMerging(*embed_dim[0:2], resolution),
            Residual(Conv2d_BN(embed_dim[1], embed_dim[1], 3, 1, 1, groups=embed_dim[1], resolution=resolution)),
            Residual(FFN(embed_dim[1], int(embed_dim[1] * 2), resolution))
        )
        conv2 = [EfficientViTBlock('s', embed_dim[1], 16, 3, attn_ratio[1], resolution, 7)
                    for i in range(3)]
        self.conv2 = nn.Sequential(*conv2)
        conv2_add = [EfficientViTBlock('s', embed_dim[1], 16, 3, attn_ratio[1], resolution, 7)
                    for i in range(3)]
        self.conv2_add = nn.Sequential(*conv2_add)

        self.conv3 = nn.Sequential(
            Residual(Conv2d_BN(embed_dim[1], embed_dim[1], 3, 1, 1, groups=embed_dim[1], resolution=resolution)),
            Residual(FFN(embed_dim[1], int(embed_dim[1] * 2), resolution)),
            PatchMerging(*embed_dim[1:3], resolution),
            Residual(Conv2d_BN(embed_dim[2], embed_dim[2], 3, 1, 1, groups=embed_dim[2], resolution=resolution)),
            Residual(FFN(embed_dim[2], int(embed_dim[2] * 2), resolution))
        )
        conv4 = [EfficientViTBlock('s', embed_dim[2], 16, 3, attn_ratio[2], resolution, 7)
                    for i in range(4)]
        self.conv4 = nn.Sequential(*conv4)
        conv4_add = [EfficientViTBlock('s', embed_dim[2], 16, 3, attn_ratio[2], resolution, 7)
                    for i in range(5)]
        self.conv4_add = nn.Sequential(*conv4_add)

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels))

        self.redir1 = MobileV2Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
        self.redir2 = MobileV2Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)

        self.visual_conv16_0 = nn.Conv2d(in_channels=192, out_channels=12, kernel_size=1)
        self.visual_conv16_1 = nn.Conv2d(in_channels=192, out_channels=12, kernel_size=1)

        self.visual_conv8_0 = nn.Conv2d(in_channels=96, out_channels=24, kernel_size=1)
        self.visual_conv8_1 = nn.Conv2d(in_channels=96, out_channels=24, kernel_size=1)

        self.visual_conv4_0 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=1)


    def forward(self, x, features_left):
        for each in self.conv0:
            each.mixer.m.resolution = x.shape[2:]
        x = self.conv0(x)
        visual_cost4 = self.visual_conv4_0(x)

        conv1 = self.conv1(x)

        for each in self.conv2:
            each.mixer.m.resolution = conv1.shape[2:]
        conv2 = self.conv2(conv1)
        visual_cost8_0 = self.visual_conv8_0(conv2)

        for each in self.conv2_add:
            each.mixer.m.resolution = conv2.shape[2:]
        conv2 = self.conv2_add(conv2)
        visual_cost8_1 = self.visual_conv8_1(conv2)

        conv3 = self.conv3(conv2)

        for each in self.conv4:
            each.mixer.m.resolution = conv3.shape[2:]
        conv4 = self.conv4(conv3)
        visual_cost16_0 = self.visual_conv16_0(conv4)

        for each in self.conv4_add:
            each.mixer.m.resolution = conv4.shape[2:]
        conv4 = self.conv4_add(conv4)
        visual_cost16_1 = self.visual_conv16_1(conv4)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6, visual_cost4, visual_cost8_0, visual_cost8_1, visual_cost16_0, visual_cost16_1


class MobileV2Residual(nn.Module):
    def __init__(self, inp, oup, stride, expanse_ratio, dilation=1):
        super(MobileV2Residual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expanse_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        pad = dilation

        self.pwconv = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, pad, dilation=dilation, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        self.pwliner = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        )

        # self.conv = nn.Sequential(
        #     nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        #     nn.BatchNorm2d(oup),
        #     nn.ReLU(inplace=True),
        # )

        # self.conv = nn.Sequential(
        #     # dw
        #     nn.Conv2d(inp, inp, 3, stride, pad, dilation=dilation, groups=inp, bias=True),
        #     # nn.BatchNorm2d(inp),
        #     nn.ReLU(inplace=True),
        #     # pw-linear
        #     nn.Conv2d(inp, oup, 1, 1, 0, bias=True),
        #     # nn.BatchNorm2d(oup),
        #     nn.ReLU(inplace=True)
        # )

        # self.seblock = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(hidden_dim, inp, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(inp, hidden_dim, 1),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        feat = self.pwconv(x)
        feat = self.dwconv(feat)
        # feat = feat * self.seblock(feat)
        feat = self.pwliner(feat)

        if self.use_res_connect:
            return x + feat
        else:
            return feat


class AttentionModule(nn.Module):
    def __init__(self, dim, img_feat_dim):
        super().__init__()
        self.conv0 = nn.Conv2d(img_feat_dim, dim, 1)

        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, cost, x):
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)
        return attn * cost