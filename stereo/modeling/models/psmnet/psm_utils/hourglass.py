# @Time    : 2023/11/10 02:27
# @Author  : zhangchenming
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from stereo.modeling.common.basic_block_3d import BasicConv3d, BasicDeconv3d


class Hourglass(nn.Module):

    def __init__(self, in_planes):
        super(Hourglass, self).__init__()

        self.conv1 = BasicConv3d(in_planes, in_planes * 2,
                                 norm_layer=nn.BatchNorm3d, act_layer=partial(nn.ReLU, inplace=True),
                                 kernel_size=3, stride=2, padding=1)

        self.conv2 = BasicConv3d(in_planes * 2, in_planes * 2,
                                 norm_layer=nn.BatchNorm3d, act_layer=None,
                                 kernel_size=3, stride=1, padding=1)

        self.conv3 = BasicConv3d(in_planes * 2, in_planes * 2,
                                 norm_layer=nn.BatchNorm3d, act_layer=partial(nn.ReLU, inplace=True),
                                 kernel_size=3, stride=2, padding=1)

        self.conv4 = BasicConv3d(in_planes * 2, in_planes * 2,
                                 norm_layer=nn.BatchNorm3d, act_layer=partial(nn.ReLU, inplace=True),
                                 kernel_size=3, stride=1, padding=1)

        self.conv5 = BasicDeconv3d(in_planes * 2, in_planes * 2,
                                   norm_layer=nn.BatchNorm3d, act_layer=None,
                                   kernel_size=3,  stride=2, padding=1, output_padding=1)

        self.conv6 = BasicDeconv3d(in_planes * 2, in_planes,
                                   norm_layer=nn.BatchNorm3d, act_layer=None,
                                   kernel_size=3,  stride=2, padding=1, output_padding=1)

    def forward(self, x, presqu=None, postsqu=None):
        out = self.conv1(x)
        pre = self.conv2(out)
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)
        out = self.conv4(out)

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)
        return out, pre, post
