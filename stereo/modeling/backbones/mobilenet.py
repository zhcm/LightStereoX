# @Time    : 2024/6/21 06:21
# @Author  : zhangchenming
import timm
import torch.nn as nn


class MobileNetV2(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        model = timm.create_model('mobilenetv2_100', pretrained=pretrained)
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = model.act1
        self.block0 = model.blocks[0]
        self.block1 = model.blocks[1]
        self.block2 = model.blocks[2]
        self.block3 = model.blocks[3:5]
        self.block4 = model.blocks[5]
        # self.block5 = model.blocks[6]

    def forward(self, images):
        c1 = self.conv_stem(images)  # [bz, 32, H/2, W/2]
        c1 = self.bn1(c1)
        c1 = self.act1(c1)
        c1 = self.block0(c1)  # [bz, 16, H/2, W/2]
        c2 = self.block1(c1)  # [bz, 24, H/4, W/4]
        c3 = self.block2(c2)  # [bz, 32, H/8, W/8]
        c4 = self.block3(c3)  # [bz, 96, H/16, W/16]
        c5 = self.block4(c4)  # [bz, 160, H/32, W/32]
        # c5 = self.block5(c5)  # [bz, 320, H/32, W/32]

        return {'scale1': c1, 'scale2': c2, 'scale3': c3, 'scale4': c4, 'scale5': c5}

    @property
    def out_dims(self):
        return {'scale1': 16, 'scale2': 24, 'scale3': 32, 'scale4': 96, 'scale5': 160}


class MobileNetV3(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        model = timm.create_model('mobilenetv3_large_100', pretrained=pretrained, features_only=True)
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = model.act1
        self.block0 = model.blocks[0]
        self.block1 = model.blocks[1]
        self.block2 = model.blocks[2]
        self.block3 = model.blocks[3:5]
        self.block4 = model.blocks[5]

    def forward(self, images):
        c1 = self.act1(self.bn1(self.conv_stem(images)))  # [bz, 16, H/2, W/2]
        c1 = self.block0(c1)  # [bz, 16, H/2, W/2]
        c2 = self.block1(c1)  # [bz, 24, H/4, W/4]
        c3 = self.block2(c2)  # [bz, 40, H/8, W/8]
        c4 = self.block3(c3)  # [bz, 112, H/16, W/16]
        c5 = self.block4(c4)  # [bz, 160, H/32, W/32]

        return {'scale1': c1, 'scale2': c2, 'scale3': c3, 'scale4': c4, 'scale5': c5}

    @property
    def out_dims(self):
        return {'scale1': 16, 'scale2': 24, 'scale3': 40, 'scale4': 112, 'scale5': 160}
