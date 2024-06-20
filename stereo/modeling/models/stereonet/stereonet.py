# @Time    : 2024/1/5 11:45
# @Author  : zhangchenming
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from stereo.modeling.common.basic_block_3d import BasicConv3d
from stereo.modeling.common.cost_volume import compute_volume
from stereo.modeling.common.disp_regression import disparity_regression

from .stereonet_utils.backbone import FeatureExtractor, ResBlock


class Refinement(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        dilations = [1, 2, 4, 8, 1, 1]
        net = [nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)]
        for dilation in dilations:
            net.append(ResBlock(channels=32, kernel_size=3, padding=dilation, dilation=dilation))

        net.append(nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        x = self.net(x)
        return x


class StereoNet(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.in_channels = cfgs.IN_CHANNELS
        self.max_disp = cfgs.MAX_DISP
        self.k = cfgs.K_DOWNSAMP_LAYERS
        self.single_refine = cfgs.SINGLE_REFINE
        self.feature = FeatureExtractor(in_channels=self.in_channels, out_channels=32, k_downsamp_layers=self.k)

        cost_agg = []
        for _ in range(4):
            cost_agg.append(BasicConv3d(in_channels=32, out_channels=32,
                                        norm_layer=nn.BatchNorm3d, act_layer=partial(nn.LeakyReLU, negative_slope=0.2),
                                        kernel_size=3, padding=1))
        cost_agg.append(nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, padding=1))
        self.cost_agg = nn.Sequential(*cost_agg)

        self.refiners = nn.ModuleList()
        for _ in range(1 if self.single_refine else self.k):
            self.refiners.append(Refinement(in_channels=self.in_channels + 1))

    def forward(self, data):
        left = data['left']
        right = data['right']

        disp_preds_left = self.get_disp_preds(left, right, side='left')
        return {'disp_pred': disp_preds_left[-1],
                'disp_preds_left': disp_preds_left}

        # disp_preds_left = self.get_disp_preds(left, right, side='left')
        # if not self.training:
        #     return {'disp_pred': disp_preds_left[-1]}
        #
        # disp_preds_right = self.get_disp_preds(right, left, side='right')
        # return {'disp_preds_left': disp_preds_left,
        #         'disp_preds_right': disp_preds_right}

    def get_disp_preds(self, reference, target, side='left'):
        reference_embedding = self.feature(reference)  # [bz, 32, H/8, W/8]
        target_embedding = self.feature(target)  # [bz, 32, H/8, W/8]

        cost = compute_volume(reference_embedding, target_embedding, maxdisp=self.max_disp // 2 ** self.k, side=side)
        cost_volume = self.cost_agg(cost)
        cost_volume = torch.squeeze(cost_volume, dim=1)

        prob = F.softmax(cost_volume, dim=1)
        disp_init = disparity_regression(prob, maxdisp=self.max_disp // 2 ** self.k)
        disp_preds = [disp_init]

        for idx, refiner in enumerate(self.refiners, start=1):
            if self.single_refine:
                scale = 2 ** self.k
            else:
                scale = 2
            disp_low = disp_preds[-1]
            new_h, new_w = int(disp_low.shape[2] * scale), int(disp_low.size()[3] * scale)

            ref_rescaled = F.interpolate(reference, [new_h, new_w], mode='bilinear', align_corners=True)
            disp_low_rescaled = F.interpolate(disp_low, [new_h, new_w], mode='bilinear', align_corners=True)
            disp_low_rescaled *= scale

            disp_refine = refiner(torch.cat([ref_rescaled, disp_low_rescaled], dim=1))
            refined_disp = F.relu(disp_refine + disp_low_rescaled)
            disp_preds.append(refined_disp)

        for i in range(len(disp_preds) - 1):
            scale = reference.shape[2] / disp_preds[i].shape[2]
            disp_preds[i] = F.interpolate(disp_preds[i], reference.shape[2:], mode='bilinear', align_corners=True)
            disp_preds[i] *= scale

        return disp_preds

    def get_loss(self, model_pred, input_data):
        disp_gt_left = input_data['disp'].unsqueeze(1)
        left_mask = (disp_gt_left < self.max_disp)
        disp_preds_left = model_pred['disp_preds_left']
        left_loss = 0
        for each_pred in disp_preds_left:
            left_loss += torch.mean(self.robust_loss(disp_gt_left[left_mask] - each_pred[left_mask], alpha=1, c=2))
        all_loss = left_loss

        # disp_gt_right = input_data['disp_right'].unsqueeze(1)
        # right_mask = (disp_gt_right < self.max_disp)
        # disp_preds_right = model_pred['disp_preds_right']
        # right_loss = 0
        # for each_pred in disp_preds_right:
        #     right_loss += torch.mean(self.robust_loss(disp_gt_right[right_mask] - each_pred[right_mask], alpha=1, c=2))
        # all_loss = (left_loss + right_loss) / 2

        tb_info = {'scalar/train/loss_disp': all_loss.item()}
        return all_loss, tb_info

    @staticmethod
    def robust_loss(x, alpha, c):
        f = (abs(alpha - 2) / alpha) * (torch.pow(torch.pow(x / c, 2) / abs(alpha - 2) + 1, alpha / 2) - 1)
        return f
