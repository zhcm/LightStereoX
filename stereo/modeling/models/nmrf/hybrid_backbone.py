# @Time    : 2024/10/14 20:06
# @Author  : zhangchenming
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from stereo.modeling.backbones.dinov2 import DINOv2
from stereo.modeling.backbones.dinov2_simple import DINOv2 as DINOv2Simple


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

        self.vits_backbone = DINOv2('vits', patch_size=14)
        url = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth'
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        state_dict.pop('mask_token')
        self.vits_backbone.load_state_dict(state_dict, strict=True)

        model = timm.create_model('mobilenetv2_100', pretrained=True)
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        # self.act1 = model.act1
        self.block0 = model.blocks[0]
        self.block1 = model.blocks[1]
        self.block2 = model.blocks[2]

        self.up1 = FPNLayer(in_channels=384, out_channels=128, high_dim=32)
        self.up2 = FPNLayer(in_channels=128, out_channels=128, high_dim=24)

        self.output_dim = 128

    def forward(self, images):
        features = self.vits_backbone.get_intermediate_layers(images, [11], reshape=True)[0]  # [bz, 384, H/14, W/14]

        c1 = self.conv_stem(images)  # [bz, 32, H/2, W/2]
        c1 = self.bn1(c1)
        # c1 = self.act1(c1)
        c1 = self.block0(c1)  # [bz, 16, H/2, W/2]
        c2 = self.block1(c1)  # [bz, 24, H/4, W/4]
        c3 = self.block2(c2)  # [bz, 32, H/8, W/8]

        s3 = self.up1(features, c3)  # [bz, 128, H/8, W/8]
        s2 = self.up2(s3, c2)  # [bz, 128, H/4, W/4]

        return s2, s3


class Hybrid2(nn.Module):
    def __init__(self):
        super().__init__()

        self.vits_backbone = DINOv2Simple('vitb', patch_size=1)
        url = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth'
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        state_dict.pop('pos_embed')
        state_dict.pop('cls_token')
        state_dict.pop('mask_token')
        state_dict.pop('patch_embed.proj.weight')
        state_dict.pop('patch_embed.proj.bias')
        self.vits_backbone.load_state_dict(state_dict, strict=False)

        model = timm.create_model('efficientnetv2_rw_m', pretrained=True)
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        # self.act1 = model.act1
        self.block0 = model.blocks[0]
        self.block1 = model.blocks[1]
        self.block2 = model.blocks[2]
        self.block3 = model.blocks[3:5]

        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=768, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(768),
            nn.SiLU()
        )

        self.up1 = FPNLayer(in_channels=768, out_channels=128, high_dim=80)
        self.up2 = FPNLayer(in_channels=128, out_channels=128, high_dim=56)

        self.output_dim = 128

    def forward(self, images):
        c1 = self.conv_stem(images)  # [bz, 32, H/2, W/2]
        c1 = self.bn1(c1)
        # c1 = self.act1(c1)
        c1 = self.block0(c1)
        c2 = self.block1(c1)  # [bz, 56, H/4, W/4]
        c3 = self.block2(c2)  # [bz, 80, H/8, W/8]
        c4 = self.block3(c3)  # [bz, 192, H/16, W/16]

        c4 = self.block4(c4)  # # [bz, 768, H/16, W/16]

        c4 = self.vits_backbone(c4)  # [bz, 768, H/16, W/16]

        s3 = self.up1(c4, c3)  # [bz, 128, H/8, W/8]  # left right cross attention
        s2 = self.up2(s3, c2)  # [bz, 128, H/4, W/4] # left right cross attention

        return s2, s3

        # efficientnet_v2, vit_base
