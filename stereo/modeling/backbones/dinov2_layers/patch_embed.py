# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

from typing import Callable, Optional, Tuple, Union

from torch import Tensor
import torch.nn as nn


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


def replace_batchnorm_with_instancenorm(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):  # 如果是2D的BatchNorm层（常用于CNN）
            setattr(model, name, nn.InstanceNorm2d(module.num_features))  # 替换为LayerNorm
        elif isinstance(module, nn.BatchNorm1d):  # 如果是1D的BatchNorm层（常用于MLP）
            setattr(model, name, nn.InstanceNorm1d(module.num_features))
        else:
            replace_batchnorm_with_instancenorm(module)  # 递归地处理子模块


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        import timm
        model = timm.create_model('efficientnetv2_rw_s', pretrained=False)
        replace_batchnorm_with_instancenorm(model)
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.block0 = model.blocks[0]
        self.block1 = model.blocks[1]
        self.block2 = model.blocks[2]
        self.block3 = model.blocks[3:5]
        self.block4 = nn.Conv2d(in_channels=160, out_channels=768, kernel_size=3, padding=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        # x = self.proj(x)  # B C H W

        c1 = self.conv_stem(x)  # [bz, 32, H/2, W/2]
        c1 = self.bn1(c1)
        c1 = self.block0(c1)
        c2 = self.block1(c1)  # [bz, 56, H/4, W/4]
        c3 = self.block2(c2)  # [bz, 80, H/8, W/8]
        c4 = self.block3(c3)  # [bz, 192, H/16, W/16]
        x = self.block4(c4)  # [bz, 768, H/16, W/16]

        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x, c3, c2

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
