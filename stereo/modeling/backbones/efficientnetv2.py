# @Time    : 2024/6/21 15:06
# @Author  : zhangchenming
import timm
import torch.nn as nn


class EfficientNetV2(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        model = timm.create_model('efficientnetv2_rw_s', pretrained=pretrained)
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = model.act1
        self.block0 = model.blocks[0]
        self.block1 = model.blocks[1]
        self.block2 = model.blocks[2]
        self.block3 = model.blocks[3:5]
        self.block4 = model.blocks[5]

    def forward(self, images):
        c1 = self.conv_stem(images)  # [bz, 24, H/2, W/2]
        c1 = self.bn1(c1)
        c1 = self.act1(c1)
        c1 = self.block0(c1)  # [bz, 24, H/2, W/2]
        c2 = self.block1(c1)  # [bz, 48, H/4, W/4]
        c3 = self.block2(c2)  # [bz, 64, H/8, W/8]
        c4 = self.block3(c3)  # [bz, 160, H/16, W/16]
        c5 = self.block4(c4)  # [bz, 272, H/32, W/32]

        return {'scale1': c1, 'scale2': c2, 'scale3': c3, 'scale4': c4, 'scale5': c5}

    @property
    def out_dims(self):
        return {'scale1': 24, 'scale2': 48, 'scale3': 64, 'scale4': 160, 'scale5': 272}
