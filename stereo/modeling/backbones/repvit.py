# @Time    : 2024/7/4 00:41
# @Author  : zhangchenming
import timm
import torch.nn as nn

"""
timm.create_model('repvit_m0_9', pretrained=False).default_cfg
timm.list_models('*repvit*')
"""


class RepVit(nn.Module):
    def __init__(self):
        super().__init__()
        model = timm.create_model('repvit_m0_9', pretrained=True)
        self.stem = model.stem
        self.stage0 = model.stages[0]
        self.stage1 = model.stages[1]
        self.stage2 = model.stages[2]
        self.stage3 = model.stages[3]

    def forward(self, images):
        c2 = self.stem(images)  # [bz, 48, H/4, W/4]
        c2 = self.stage0(c2)  # [bz, 48, H/4, W/4]
        c3 = self.stage1(c2)  # [bz, 96, H/8, W/8]
        c4 = self.stage2(c3)  # [bz, 192, H/16, W/16]
        c5 = self.stage3(c4)  # [bz, 384, H/32, W/32]

        return [c2, c3, c4, c5]

    @property
    def out_dims(self):
        return [48, 96, 192, 384]
