# @Time    : 2024/10/14 20:06
# @Author  : zhangchenming
import torch
import torch.nn as nn
import torch.nn.functional as F
from stereo.modeling.backbones.dinov2 import DINOv2

import xformers.ops


class MemoryEfficientCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, qkv_bias=False):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=qkv_bias)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op = None

    def forward(self, x, context=None):
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class FPNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, high_dim):
        super(FPNLayer, self).__init__()
        self.deconv = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU())

        self.conv = nn.Sequential(nn.Conv2d(out_channels + high_dim, out_channels, kernel_size=3, padding=1, bias=False),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU())

    def forward(self, low, high):
        low = self.deconv(low)
        if low.shape != high.shape:
            low = F.interpolate(low, size=(high.shape[-2], high.shape[-1]), mode='nearest')
        feat = torch.cat([high, low], 1)
        feat = self.conv(feat)
        return feat


class Hybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.vits_backbone = DINOv2('vitb', patch_size=16)
        state_dict = torch.hub.load_state_dict_from_url('teacher_checkpoint.pth', map_location="cpu")
        new_state_dict = {}
        for each in state_dict['teacher']:
            if 'backbone' in each:
                new_state_dict[each[9:]] = state_dict['teacher'][each]
        new_state_dict.pop('mask_token')
        # new_state_dict.pop('pos_embed')
        self.vits_backbone.load_state_dict(new_state_dict, strict=True)

        self.up1 = FPNLayer(in_channels=768, out_channels=128, high_dim=64)
        self.up2 = FPNLayer(in_channels=128, out_channels=128, high_dim=48)

        self.output_dim = 128

        # self.attn1 = MemoryEfficientCrossAttention(query_dim=128, context_dim=128)
        # self.attn2 = MemoryEfficientCrossAttention(query_dim=128, context_dim=128)

    def forward(self, images):
        c4, c3, c2 = self.vits_backbone.get_intermediate_layers(images, reshape=True)  # [bz, 768, H/16, W/16]

        s3 = self.up1(c4, c3)  # [bz, 128, H/8, W/8]  # left right cross attention

        # left_feature, right_feature = torch.chunk(s3, 2, dim=0)
        # B, _, w, h = left_feature.shape
        # x = left_feature.flatten(2).transpose(1, 2)
        # context = right_feature.flatten(2).transpose(1, 2)
        # x = self.attn1(x, context=context) + x
        # left_feature = x.reshape(B, w, h, -1).permute(0, 3, 1, 2).contiguous()
        # s3 = torch.concatenate([left_feature, right_feature], dim=0)

        s2 = self.up2(s3, c2)  # [bz, 128, H/4, W/4] # left right cross attention

        # left_feature, right_feature = torch.chunk(s2, 2, dim=0)
        # B, _, w, h = left_feature.shape
        # x = left_feature.flatten(2).transpose(1, 2)
        # context = right_feature.flatten(2).transpose(1, 2)
        # x = self.attn2(x, context=context) + x
        # left_feature = x.reshape(B, w, h, -1).permute(0, 3, 1, 2).contiguous()
        # s2 = torch.concatenate([left_feature, right_feature], dim=0)

        return s2, s3
