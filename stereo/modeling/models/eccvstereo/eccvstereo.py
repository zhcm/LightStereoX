import torch
import torch.nn as nn
import torch.nn.functional as F

from stereo.modeling.common.basic_block_2d import BasicConv2d
from stereo.modeling.common.cost_volume import correlation_volume, build_gwc_volume, CoExCostVolume, build_sub_volume
from stereo.modeling.common.disp_regression import disparity_regression, topkpool_regression
from stereo.modeling.common.refinement import context_upsample, StereoNetRefinement

from stereo.modeling.models.coex.aggregation import Aggregation
from .eccv_utils.backbone import Feature
from .eccv_utils.hourglass import Hourglass2D


class StereoBase(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.max_disp = cfgs.MAX_DISP

        self.feature = Feature()
        self.cost_agg = Hourglass2D(self.max_disp // 4)
        # self.cost_agg = Aggregation(in_channels=1, gce=False)

        # upsample
        self.spx_4 = nn.Sequential(
            BasicConv2d(48, 24, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(24, 24, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU))
        self.spx = nn.ConvTranspose2d(24, 9, kernel_size=4, stride=4, padding=0)


    def forward(self, data):
        image1 = data['left']
        image2 = data['right']

        features_left = self.feature(image1)
        features_right = self.feature(image2)

        cost_volume = correlation_volume(features_left, features_right, max_disp=self.max_disp // 4)
        cost_volume = self.cost_agg(cost_volume)  # [bz, max_disp/4, H/4, W/4]
        # cost_volume = cost_volume.unsqueeze(1)
        # cost_volume = build_gwc_volume(features_left, features_right, maxdisp=self.max_disp // 4, num_groups=8)
        # cost_volume = self.cost_agg(features_left=None, cost=cost_volume).squeeze(1)

        init_disp = topkpool_regression(cost_volume, k=2)
        # prob = F.softmax(cost_volume, dim=1)
        # init_disp = disparity_regression(prob, self.max_disp // 4)  # [bz, 1, H/4, W/4]

        xspx = self.spx_4(features_left)  # [bz, 24, H/4, W/4]
        spx_pred = self.spx(xspx)  # [bz, 9, H, W]
        spx_pred = F.softmax(spx_pred, 1)  # [bz, 9, H, W]
        disp_pred = context_upsample(init_disp * 4., spx_pred.float()).unsqueeze(1)  # [bz, 1, H, W]

        result = {'disp_pred': disp_pred}

        if self.training:
            init_disp = F.interpolate(init_disp, image1.shape[2:], mode='bilinear', align_corners=True)
            init_disp *= 4
            result['init_disp'] = init_disp

        return result


    def get_loss(self, model_pred, input_data):
        # disp_pred = model_pred['disp_pred']
        # disp_gt = input_data["disp"]  # [bz, h, w]
        # mask = (disp_gt < self.max_disp) & (disp_gt > 0)  # [bz, h, w]
        # valid = mask.float()  # [bz, h, w]
        #
        # disp_gt = disp_gt.unsqueeze(1)  # [bz, 1, h, w]
        # mag = torch.sum(disp_gt ** 2, dim=1).sqrt()  # [bz, h, w]
        # valid = ((valid >= 0.5) & (mag < self.max_disp)).unsqueeze(1)  # [bz, 1, h, w]
        # assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
        # assert not torch.isinf(disp_gt[valid.bool()]).any()
        #
        # disp_loss = 1.0 * F.smooth_l1_loss(disp_pred[valid.bool()], disp_gt[valid.bool()], reduction='mean')
        #
        # init_disp = model_pred['init_disp']
        # disp_loss += 0.5 * F.smooth_l1_loss(init_disp[valid.bool()], disp_gt[valid.bool()], reduction='mean')
        #
        # loss_info = {'scalar/train/loss_disp': disp_loss.item()}
        # return disp_loss, loss_info

        disp_gt = input_data["disp"]  # [bz, h, w]
        disp_gt = disp_gt.unsqueeze(1)  # [bz, 1, h, w]
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)  # [bz, 1, h, w]

        disp_pred = model_pred['disp_pred']
        loss = 1.0 * F.smooth_l1_loss(disp_pred[mask], disp_gt[mask], reduction='mean')

        init_disp = model_pred['init_disp']
        loss += 0.3 * F.smooth_l1_loss(init_disp[mask], disp_gt[mask], reduction='mean')

        loss_info = {'scalar/train/loss_disp': loss.item()}
        return loss, loss_info
