import torch
import torch.nn as nn
import torch.nn.functional as F

from stereo.modeling.common.basic_block_2d import BasicDeconv2d
from timm.models.layers import DropPath


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  # depthwise conv
        # self.norm = LayerNorm(dim, eps=1e-6)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class Aggregation(nn.Module):
    def __init__(self, in_channels=48):
        super(Aggregation, self).__init__()

        depths = [6, 8, 10]
        self.num_stages = 3
        dims = [48, 48, 96, 192]

        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()

        layer_scale_init_value = 1e-6
        drop_path_rate = 0.1
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(self.num_stages):
            if i == 0:
                downsample_layer = LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
            else:
                downsample_layer = nn.Sequential(
                        LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                        nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2))

            self.downsample_layers.append(downsample_layer)

            stage = nn.Sequential(
                *[Block(dim=dims[i+1], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = LayerNorm(dims[-1], eps=1e-6, data_format="channels_first")

        self.conv_t1 = BasicDeconv2d(in_channels=192, out_channels=96, kernel_size=4,stride=2,padding=1)
        self.conv_t2 = BasicDeconv2d(in_channels=96, out_channels=48, kernel_size=4,stride=2,padding=1)
        self.redir1 = nn.Sequential(LayerNorm(dims[-2], eps=1e-6, data_format="channels_first"),
                                    Block(96))
        self.redir2 = nn.Sequential(LayerNorm(dims[-3], eps=1e-6, data_format="channels_first"),
                                    Block(48))

        self.norm_last = LayerNorm(dims[0], eps=1e-6, data_format="channels_first")

    def forward(self, x):
        outs = []
        for i in range(3):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outs.append(x)

        outs[2] = self.norm(outs[2])
        x = F.gelu(self.conv_t1(outs[2]) + self.redir1(outs[1]))
        x = F.gelu(self.conv_t2(x) + self.redir2(outs[0]))

        x = self.norm_last(x)

        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x