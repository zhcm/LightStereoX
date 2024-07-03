# @Time    : 2023/11/27 15:39
# @Author  : zhangchenming
import torch
from functools import partial
from torch import nn, Tensor
from torchvision.models.densenet import _DenseBlock
from .utils import center_crop
from stereo.modeling.models.lightfast.basic_block_2d import BasicConv2d


class TransitionUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale: int = 2):
        super().__init__()
        if scale == 2:
            self.convTrans = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                                kernel_size=3, stride=2)
        elif scale == 4:
            self.convTrans = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=3, stride=2, bias=False),
                nn.InstanceNorm2d(out_channels),
                nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels,
                                   kernel_size=3, stride=2)
            )

    def forward(self, x: Tensor, skip: Tensor):
        out = self.convTrans(x)
        out = center_crop(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out


class Tokenizer(nn.Module):
    def __init__(self, block_config: list, backbone_feat_channel: list, hidden_dim: int, growth_rate: int):
        super(Tokenizer, self).__init__()

        self.num_resolution = len(backbone_feat_channel)
        self.block_config = block_config
        self.growth_rate = growth_rate

        self.bottle_neck = _DenseBlock(block_config[0], backbone_feat_channel[0],
                                       4, drop_rate=0.0,
                                       growth_rate=growth_rate, memory_efficient=True)
        for name, layer in self.bottle_neck.items():
            layer.norm1 = nn.InstanceNorm2d(layer.norm1.num_features)
            layer.norm2 = nn.InstanceNorm2d(layer.norm2.num_features)

        up = []
        dense_block = []
        prev_block_channels = growth_rate * block_config[0]
        for i in range(self.num_resolution):
            if i == self.num_resolution - 1:
                up.append(TransitionUp(prev_block_channels, hidden_dim, 4))
                tmp_block = nn.Sequential(
                    BasicConv2d(in_channels=hidden_dim + 3, out_channels=hidden_dim,
                                norm_layer=nn.InstanceNorm2d, act_layer=partial(nn.ReLU, inplace=True),
                                kernel_size=3, padding=1),
                    BasicConv2d(in_channels=hidden_dim, out_channels=hidden_dim,
                                norm_layer=nn.InstanceNorm2d, act_layer=partial(nn.ReLU, inplace=True),
                                kernel_size=3, padding=1),
                )
                dense_block.append(tmp_block)

            else:
                up.append(TransitionUp(prev_block_channels, prev_block_channels, 2))
                cur_channels_count = prev_block_channels + backbone_feat_channel[i + 1]
                tmp_block = _DenseBlock(block_config[i + 1], cur_channels_count,
                                        4, drop_rate=0.0, growth_rate=growth_rate,
                                        memory_efficient=True)
                for name, layer in tmp_block.items():
                    layer.norm1 = nn.InstanceNorm2d(layer.norm1.num_features)
                    layer.norm2 = nn.InstanceNorm2d(layer.norm2.num_features)

                dense_block.append(tmp_block)
                prev_block_channels = growth_rate * block_config[i + 1]

        self.up = nn.ModuleList(up)
        self.dense_block = nn.ModuleList(dense_block)

    def forward(self, features: list):
        """
        :param features:
            0: [2bz, 128, H//16, W//16]
            1: [2bz, 128, H//8, W//8]
            2: [2bz, 64, H//4, W//4]
            3: [2bz, 3, H, W]
        :return: feature descriptor at full resolution [2bz, 128, H, W]
        """
        output = self.bottle_neck(features[0])  # [2bz, 144, H//16, W//16]
        output = output[:, -(self.block_config[0] * self.growth_rate):]  # # [2bz, 16, H//16, W//16] take only the new features

        for i in range(self.num_resolution):
            hs = self.up[i](output, features[i + 1])  # scale up and concat
            output = self.dense_block[i](hs)  # denseblock

            if i < self.num_resolution - 1:  # other than the last convolution block
                output = output[:, -(self.block_config[i + 1] * self.growth_rate):]  # take only the new features

        return output
