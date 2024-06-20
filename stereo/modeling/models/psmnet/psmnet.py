# @Time    : 2023/9/3 10:29
# @Author  : zhangchenming
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from stereo.modeling.common.basic_block_3d import BasicConv3d
from stereo.modeling.common.cost_volume import build_concat_volume
from .psm_utils.backbone import PsmnetBackbone
from .psm_utils.hourglass import Hourglass
from .psm_utils.soft_argmin import FasterSoftArgmin


class PSMNet(nn.Module):
    def __init__(self, cfgs):
        super(PSMNet, self).__init__()
        self.max_disp = cfgs.MAX_DISP

        # backbone
        self.feature_extraction = PsmnetBackbone()

        # aggregation
        self.dres0 = nn.Sequential(BasicConv3d(in_channels=64, out_channels=32,
                                               norm_layer=nn.BatchNorm3d,
                                               act_layer=partial(nn.ReLU, inplace=True),
                                               kernel_size=3, stride=1, padding=1),
                                   BasicConv3d(in_channels=32, out_channels=32,
                                               norm_layer=nn.BatchNorm3d,
                                               act_layer=partial(nn.ReLU, inplace=True),
                                               kernel_size=3, stride=1, padding=1))

        self.dres1 = nn.Sequential(BasicConv3d(in_channels=32, out_channels=32,
                                               norm_layer=nn.BatchNorm3d,
                                               act_layer=partial(nn.ReLU, inplace=True),
                                               kernel_size=3, stride=1, padding=1),
                                   BasicConv3d(in_channels=32, out_channels=32,
                                               norm_layer=nn.BatchNorm3d,
                                               act_layer=None,
                                               kernel_size=3, stride=1, padding=1))

        self.dres2 = Hourglass(in_planes=32)
        self.dres3 = Hourglass(in_planes=32)
        self.dres4 = Hourglass(in_planes=32)

        # classify
        self.classify1 = nn.Sequential(BasicConv3d(32, 32,
                                                   norm_layer=nn.BatchNorm3d,
                                                   act_layer=partial(nn.ReLU, inplace=True),
                                                   kernel_size=3, stride=1, padding=1),
                                       nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classify2 = nn.Sequential(BasicConv3d(32, 32,
                                                   norm_layer=nn.BatchNorm3d,
                                                   act_layer=partial(nn.ReLU, inplace=True),
                                                   kernel_size=3, stride=1, padding=1),
                                       nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classify3 = nn.Sequential(BasicConv3d(32, 32,
                                                   norm_layer=nn.BatchNorm3d,
                                                   act_layer=partial(nn.ReLU, inplace=True),
                                                   kernel_size=3, stride=1, padding=1),
                                       nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        # disparity_regression
        self.disp_reg = FasterSoftArgmin(self.max_disp)

    def forward(self, data):
        left = data['left']
        right = data['right']
        left_feats = self.feature_extraction(left)
        right_feats = self.feature_extraction(right)

        raw_cost = build_concat_volume(left_feats, right_feats, self.max_disp // 4)

        cost0 = self.dres0(raw_cost)
        cost0 = self.dres1(cost0) + cost0
        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0
        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0
        out3, pre3, post3 = self.dres4(out2, pre2, post2)
        out3 = out3 + cost0

        cost1 = self.classify1(out1)
        cost2 = self.classify2(out2) + cost1
        cost3 = self.classify3(out3) + cost2

        cost1 = F.interpolate(cost1, [self.max_disp, left.shape[2], left.shape[3]],
                              mode='trilinear', align_corners=True)
        cost2 = F.interpolate(cost2, [self.max_disp, left.shape[2], left.shape[3]],
                              mode='trilinear', align_corners=True)
        cost3 = F.interpolate(cost3, [self.max_disp, left.shape[2], left.shape[3]],
                              mode='trilinear', align_corners=True)

        cost1 = torch.squeeze(cost1, 1)
        cost2 = torch.squeeze(cost2, 1)
        cost3 = torch.squeeze(cost3, 1)

        disp1 = self.disp_reg(cost1)
        disp2 = self.disp_reg(cost2)
        disp3 = self.disp_reg(cost3)

        return {'disp_pred': disp3,
                'disp_preds': [disp1, disp2, disp3]}

    def get_loss(self, model_pred, input_data):
        pass
