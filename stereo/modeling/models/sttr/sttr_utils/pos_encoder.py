# @Time    : 2023/11/27 15:42
# @Author  : zhangchenming
import torch
from torch import nn


class PositionEncodingSine1DRelative(nn.Module):
    """
    relative sine encoding 1D, partially inspired by DETR (https://github.com/facebookresearch/detr)
    """
    def __init__(self, num_pos_feats=64, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    @torch.no_grad()
    def forward(self, w, scale, device):
        """
        :param device:
        :param w:
        :param scale:
        :return: pos encoding [2w-1,C]
        """
        # 从 w-1 到 -(w-1) populate all possible relative distances
        x_embed = torch.linspace(w - 1, -w + 1, 2 * w - 1, dtype=torch.float32, device=device)
        x_embed = x_embed * scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)  # [num_pos_feats]

        pos_x = x_embed[:, None] / dim_t  # [2w-1, num_pos_feats]
        # interleave cos and sin instead of concatenate
        pos = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)  # [2w-1, num_pos_feats]

        return pos
