# @Time    : 2024/3/11 10:21
# @Author  : zhangchenming
import torch
import torch.nn as nn
import torch.nn.functional as F

from stereo.modeling.common.basic_block_2d import BasicConv2d
from stereo.modeling.common.cost_volume import CoExCostVolume
from stereo.modeling.common.disp_regression import topkpool_regression
from stereo.modeling.common.refinement import context_upsample

from .backbone import CoExBackbone, Conv2x
from .aggregation import Aggregation


class CoEx(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.max_disp = cfgs.MAX_DISP

        self.backbone = CoExBackbone()

        self.pre_cost_conv = nn.Sequential(
            BasicConv2d(96, 48, kernel_size=3, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU),
            nn.Conv2d(48, 48, kernel_size=1))

        self.costVolume = CoExCostVolume(self.max_disp // 4)

        self.aggregation = Aggregation()

        self.spx_4 = nn.Sequential(
            BasicConv2d(96, 24, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(24, 24, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU))
        self.spx_2 = Conv2x(24, 32)
        self.spx = nn.ConvTranspose2d(64, 9, kernel_size=4, stride=2, padding=1)

    def forward(self, data):
        image1 = data['left']
        image2 = data['right']

        features_left, stem_2x_left = self.backbone(image1)
        features_right, _ = self.backbone(image2)

        match_left = self.pre_cost_conv(features_left[0])
        match_right = self.pre_cost_conv(features_right[0])
        match_left = match_left / torch.norm(match_left, p=2, dim=1, keepdim=True)
        match_right = match_right / torch.norm(match_right, p=2, dim=1, keepdim=True)

        cost = self.costVolume(match_left, match_right)
        cost = cost[:, :, :-1, :, :]  # [bz, num_group, max_disp/4, H/4, W/4]
        cost = self.aggregation(features_left, cost)  # [bz, 1, max_disp/4, H/4, W/4]

        disp_4 = topkpool_regression(cost.squeeze(1), k=2)  # [bz, 1, H/4, W/4]

        xspx = self.spx_4(features_left[0])
        xspx = self.spx_2(xspx, stem_2x_left)
        spx_pred = self.spx(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        disp_pred = context_upsample(disp_4 * 4, spx_pred).unsqueeze(1)  # [bz, 1, H, W]
        result = {'disp_pred': disp_pred}
        if self.training:
            disp_4 = F.interpolate(disp_4, image1.shape[2:], mode='bilinear', align_corners=False)
            disp_4 *= 4
            result['disp_4'] = disp_4

        return result

    def get_loss(self, model_pred, input_data):
        disp_gt = input_data["disp"]  # [bz, h, w]
        disp_gt = disp_gt.unsqueeze(1)  # [bz, 1, h, w]
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)  # [bz, 1, h, w]

        disp_pred = model_pred['disp_pred']
        loss = 1.0 * F.smooth_l1_loss(disp_pred[mask], disp_gt[mask], reduction='mean')

        disp_4 = model_pred['disp_4']
        loss += 0.3 * F.smooth_l1_loss(disp_4[mask], disp_gt[mask], reduction='mean')

        loss_info = {'scalar/train/loss_disp': loss.item()}
        return loss, loss_info
