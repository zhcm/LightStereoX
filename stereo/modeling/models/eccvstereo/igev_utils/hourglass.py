import torch
import torch.nn as nn
from stereo.modeling.common.basic_block_3d import BasicConv3d, BasicDeconv3d
from .igev_blocks import FeatureAtt


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

        self.feature_att_8 = FeatureAtt(in_channels * 2, 64)
        self.feature_att_16 = FeatureAtt(in_channels * 4, 192)
        self.feature_att_32 = FeatureAtt(in_channels * 6, 160)
        self.feature_att_up_16 = FeatureAtt(in_channels * 4, 192)
        self.feature_att_up_8 = FeatureAtt(in_channels * 2, 64)

    def forward(self, x, features):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)

        return conv
