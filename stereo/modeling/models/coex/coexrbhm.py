# @Time    : 2024/4/2 10:53
# @Author  : zhangchenming
import torch
import torch.nn as nn
import torch.nn.functional as F
from .coex_backbone import CoExBackbone
from .coex_cost_processor import CoExCostProcessor
from .coex_disp_processor import CoExDispProcessor
from functools import partial
from .rbhm import Refinement


class CoExHeight(nn.Module):
    def __init__(self, rbhm_pretrained=''):
        super().__init__()
        self.max_disp = 192
        spixel_branch_channels = [32, 48]
        chans = [16, 24, 32, 96, 160]
        matching_weighted = False
        matching_head = 1
        gce = True
        aggregation_disp_strides = 2
        aggregation_channels = [16, 32, 48]
        aggregation_blocks_num = [2, 2, 2]
        regression_topk = 2

        self.Backbone = CoExBackbone(spixel_branch_channels=spixel_branch_channels)
        self.CostProcessor = CoExCostProcessor(max_disp=self.max_disp,
                                               gce=gce,
                                               matching_weighted=matching_weighted,
                                               spixel_branch_channels=spixel_branch_channels,
                                               matching_head=matching_head,
                                               aggregation_disp_strides=aggregation_disp_strides,
                                               aggregation_channels=aggregation_channels,
                                               aggregation_blocks_num=aggregation_blocks_num,
                                               chans=chans)
        self.DispProcessor = CoExDispProcessor(max_disp=self.max_disp, regression_topk=regression_topk, chans=chans)

        self.height_head = Refinement(in_channels=4)
        pretrained_state_dict = torch.load(rbhm_pretrained, map_location='cpu')
        state_dict = {}
        for key, val in pretrained_state_dict.items():
            state_dict[key.replace('height_head.', '')] = val
        self.height_head.load_state_dict(state_dict)

    def forward(self, inputs):
        with torch.no_grad():
            """Forward the network."""
            backbone_out = self.Backbone(inputs)
            inputs.update(backbone_out)
            cost_out = self.CostProcessor(inputs)
            inputs.update(cost_out)
            disp_out = self.DispProcessor(inputs)

        if self.training:
            pred_height = self.height_head(torch.cat([inputs["left"], disp_out['disp_ests'][0].unsqueeze(1)], dim=1))
            pred_height = torch.sigmoid(pred_height) * 10
            return {'disp_preds': disp_out['disp_ests'],
                    'disp_pred': disp_out['disp_ests'][0],
                    'pred_height': pred_height.squeeze(1)}
        else:
            pred_height = self.height_head(torch.cat([inputs["left"], disp_out['inference_disp']['disp_est'].unsqueeze(1)], dim=1))
            pred_height = torch.sigmoid(pred_height) * 10
            return {'disp_pred': disp_out['inference_disp']['disp_est'],
                    'pred_height': pred_height.squeeze(1)}

    def get_loss(self, model_preds, input_data):
        disp_gt = input_data["disp"]  # [bz, h, w]
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)  # [bz, h, w]
        dilated_bump_mask = input_data['dilated_bump_mask']
        bump_mask = input_data['bump_mask']

        # weights = [1.0, 0.3]
        # loss = 0.0
        # for disp_est, weight in zip(model_preds['disp_preds'], weights):
        #     loss += weight * F.smooth_l1_loss(disp_est[mask & dilated_bump_mask], disp_gt[mask & dilated_bump_mask], size_average=True)
        # loss = loss * 0.77
        # loss_info = {'scalar/train/loss_disp': loss.item()}

        height_loss = F.smooth_l1_loss(model_preds['pred_height'][bump_mask], input_data['bump_height_map'][bump_mask], size_average=True)
        loss_info = {'scalar/train/loss_height': height_loss.item()}
        loss = height_loss

        return loss, loss_info
