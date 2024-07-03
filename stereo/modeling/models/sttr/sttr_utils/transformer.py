# @Time    : 2023/11/27 15:44
# @Author  : zhangchenming
from typing import Optional

import copy
import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
from .attention import MultiheadAttentionRelative


class Transformer(nn.Module):
    def __init__(self, hidden_dim: int = 128, nhead: int = 8, num_attn_layers: int = 6):
        super().__init__()
        self.num_attn_layers = num_attn_layers

        self_attn_layer = TransformerSelfAttnLayer(hidden_dim, nhead)
        self.self_attn_layers = nn.ModuleList([copy.deepcopy(self_attn_layer) for i in range(num_attn_layers)])

        cross_attn_layer = TransformerCrossAttnLayer(hidden_dim, nhead)
        self.cross_attn_layers = nn.ModuleList([copy.deepcopy(cross_attn_layer) for i in range(num_attn_layers)])

    def forward(self, feat_left: torch.Tensor, feat_right: torch.Tensor, pos_enc: Optional[Tensor] = None):
        """
        :param feat_left: [N,C,H/downsample,W/downsample]
        :param feat_right: [N,C,H/downsample,W/downsample]
        :param pos_enc:  [W/downsample, W/downsample, C]
        :return: [N,H/downsample,W/downsample,W/downsample], dim=2 is left image, dim=3 is right image
        """
        bz, c, h, w = feat_left.shape

        # [c, w, h, bz] -> [c, w, h*bz] -> [w, h*bz, c]
        feat_left = feat_left.permute(1, 3, 2, 0).flatten(2).permute(1, 2, 0)
        feat_right = feat_right.permute(1, 3, 2, 0).flatten(2).permute(1, 2, 0)

        attn_weight = None
        for idx, (self_attn, cross_attn) in enumerate(zip(self.self_attn_layers, self.cross_attn_layers)):
            # self attention
            feat_left, feat_right = checkpoint(self_attn, feat_left, feat_right, pos_enc)
            feat_left, feat_right, attn_weight = checkpoint(cross_attn, feat_left, feat_right, pos_enc,
                                                            idx == self.num_attn_layers - 1)

        # [h*bz, w, w] -> [h, bz, w, w] -> [bz, h, w, w]
        attn_weight = attn_weight.view(h, bz, w, w).permute(1, 0, 2, 3)
        return attn_weight


class TransformerSelfAttnLayer(nn.Module):
    def __init__(self, hidden_dim: int, nhead: int):
        super().__init__()
        self.attn = MultiheadAttentionRelative(hidden_dim, nhead)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, feat_left: Tensor, feat_right: Tensor, pos: Optional[Tensor] = None):
        """
        :param feat_left: [w, h*bz, c]
        :param feat_right: [w, h*bz, c]
        :param pos: [w, w, c]
        :return: updated image feature
        """
        feature = torch.cat([feat_left, feat_right], dim=1)  # [w, 2*h*bz, c]
        feat2 = self.norm(feature)
        feat2, _, _ = self.attn(query=feat2, key=feat2, value=feat2, pos_enc=pos)
        feat = feature + feat2
        feat_left, feat_right = feat.chunk(2, dim=1)
        return feat_left, feat_right


class TransformerCrossAttnLayer(nn.Module):
    def __init__(self, hidden_dim: int, nhead: int):
        super().__init__()
        self.cross_attn = MultiheadAttentionRelative(hidden_dim, nhead)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, feat_left: Tensor, feat_right: Tensor, pos: Optional[Tensor] = None,
                last_layer: Optional[bool] = False):
        """
        :param feat_left: [w,h*bz,C]
        :param feat_right: [w,h*bz,C]
        :param pos: [w, w, C]
        :param last_layer:
        :return: update image feature and attention weight
        """
        if pos is not None:
            pos_flipped = pos.transpose(0, 1)
        else:
            pos_flipped = pos

        if last_layer:
            w = feat_left.size(0)
            attn_mask = self.generate_square_subsequent_mask(w).to(feat_left.device)
        else:
            attn_mask = None

        feat_left_2 = self.norm1(feat_left)
        feat_right_2 = self.norm1(feat_right)
        feat_right_2, _, _ = self.cross_attn(query=feat_right_2,
                                             key=feat_left_2,
                                             value=feat_left_2,
                                             pos_enc=pos_flipped)
        feat_right = feat_right + feat_right_2
        feat_right_2 = self.norm2(feat_right)

        feat_left_2, _, raw_attn = self.cross_attn(query=feat_left_2,
                                                   key=feat_right_2,
                                                   value=feat_right_2,
                                                   attn_mask=attn_mask,
                                                   pos_enc=pos)
        feat_left = feat_left + feat_left_2
        return feat_left, feat_right, raw_attn

    @torch.no_grad()
    def generate_square_subsequent_mask(self, sz: int):
        """
        Generate a mask which is upper triangular
        :param sz: square matrix size
        :return: diagonal binary mask [sz,sz]
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask[mask == 1] = float('-inf')
        return mask
