import torch
import torch.nn as nn
import torch.nn.functional as F

from stereo.modeling.common.basic_block_2d import BasicConv2d, BasicDeconv2d
from stereo.modeling.common.basic_block_3d import BasicConv3d, BasicDeconv3d
from .backbone import FPNLayer
from stereo.modeling.models.lightstereo.aggregation import ChannelAtt


def drop_path(x: torch.Tensor,
              drop_prob: float = 0.,
              training: bool = False) -> torch.Tensor:
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # handle tensors with different dimensions, not just 4D tensors.
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv3d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)

        self.conv0_1 = nn.Conv3d(dim, dim, (1, 1, 7), padding=(0, 0, 3), groups=dim)
        self.conv0_2 = nn.Conv3d(dim, dim, (1, 7, 1), padding=(0, 3, 0), groups=dim)
        self.conv0_3 = nn.Conv3d(dim, dim, (3, 1, 1), padding=(1, 0, 0), groups=dim)

        self.conv1_1 = nn.Conv3d(dim, dim, (1, 1, 11), padding=(0, 0, 5), groups=dim)
        self.conv1_2 = nn.Conv3d(dim, dim, (1, 11, 1), padding=(0, 5, 0), groups=dim)
        self.conv1_3 = nn.Conv3d(dim, dim, (3, 1, 1), padding=(1, 0, 0), groups=dim)

        self.conv2_1 = nn.Conv3d(dim, dim, (1, 1, 21), padding=(0, 0, 10), groups=dim)
        self.conv2_2 = nn.Conv3d(dim, dim, (1, 21, 1), padding=(0, 10, 0), groups=dim)
        self.conv2_3 = nn.Conv3d(dim, dim, (3, 1, 1), padding=(1, 0, 0), groups=dim)

        self.conv3 = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)
        attn_0 = self.conv0_3(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
        attn_1 = self.conv1_3(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn_2 = self.conv2_3(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u


class SpatialAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_1 = nn.Conv3d(dim, dim, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(dim)
        self.proj_2 = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path_rate=0., act_layer=nn.GELU):
        super().__init__()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.norm1 = nn.BatchNorm3d(dim)
        self.attn = SpatialAttention(dim)

        self.norm2 = nn.BatchNorm3d(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        feat = self.norm1(x)
        feat = self.attn(feat)
        feat = self.layer_scale_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * feat
        self.drop_path(feat)
        feat = feat + x

        feat2 = self.norm2(feat)
        feat2 = self.mlp(feat2)
        feat2 = self.layer_scale_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * feat2
        feat2 = self.drop_path(feat2)
        feat2 = feat2 + feat

        return feat2


class Aggregation(nn.Module):
    def __init__(self, in_channels=48):
        super(Aggregation, self).__init__()

        depths = [2, 2, 2]
        self.num_stages = 3
        embed_dims = [8, 16, 32]
        mlp_ratios = [2, 2, 2]
        drop_path_rate = 0.1

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0

        for i in range(self.num_stages):
            patch_embed = BasicConv3d(in_channels=8 if i == 0 else embed_dims[i - 1], out_channels=embed_dims[i],
                                      kernel_size=3, stride=1 if i == 0 else 2, padding=1,
                                      norm_layer=nn.BatchNorm3d)

            block = nn.ModuleList([Block(dim=embed_dims[i], mlp_ratio=mlp_ratios[i],
                                         drop=0.0, drop_path_rate=dpr[cur + j])
                                   for j in range(depths[i])])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        self.conv_up_2 = BasicDeconv3d(embed_dims[2], embed_dims[1], kernel_size=4, stride=2, padding=1,
                                       norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU)
        self.conv_up_1 = BasicDeconv3d(embed_dims[1], embed_dims[0], kernel_size=4, stride=2, padding=1,
                                       norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU)

        self.conv_agg_2 = nn.Sequential(
            BasicConv3d(embed_dims[1] * 2, embed_dims[1], kernel_size=1,
                        norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU),
            BasicConv3d(embed_dims[1], embed_dims[1], kernel_size=3, padding=1,
                        norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU))

        self.conv_agg_1 = nn.Sequential(
            BasicConv3d(embed_dims[0] * 2, embed_dims[0], kernel_size=1,
                        norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU),
            BasicConv3d(embed_dims[0], embed_dims[0], kernel_size=3, padding=1,
                        norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU))

        self.att2 = ChannelAtt(embed_dims[2], 96)
        self.att1 = ChannelAtt(embed_dims[1], 32)
        self.att0 = ChannelAtt(embed_dims[0], 24)

        self.last_conv = nn.Conv3d(embed_dims[0], 1, 3, 1, 1, bias=False)

    def forward(self, x, feat_left):
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x = patch_embed(x)
            for blk in block:
                x = blk(x)

            B, C, D, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()

            outs.append(x)

        x = self.att2(outs[2], feat_left[2])
        x = self.conv_up_2(x)
        x = torch.cat([x, outs[1]], 1)
        x = self.conv_agg_2(x)

        x = self.att1(x, feat_left[1])
        x = self.conv_up_1(x)
        x = torch.cat([x, outs[0]], 1)
        x = self.conv_agg_1(x)

        x = self.att0(x, feat_left[0])
        x = self.last_conv(x)

        return x


if __name__ == '__main__':
    MSCAN()(torch.randn(1, 48, 256, 256))
