# @Time    : 2024/4/2 12:31
# @Author  : zhangchenming
import torch
import torch.nn as nn
import torch.nn.functional as F
from .psmnet_backbone import PSMNet as PSMNetBackbone
from .psmnet_cost_processor import PSMCostProcessor
from .psmnet_disp_processor import PSMDispProcessor


class PSMNet(nn.Module):
    def __init__(self, max_disp):
        super().__init__()
        self.maxdisp = max_disp

        self.Backbone = PSMNetBackbone()
        self.CostProcessor = PSMCostProcessor(max_disp=self.maxdisp)
        self.DispProcessor = PSMDispProcessor(max_disp=self.maxdisp)

    def forward(self, inputs):
        """Forward the network."""
        backbone_out = self.Backbone(inputs)
        inputs.update(backbone_out)
        cost_out = self.CostProcessor(inputs)
        inputs.update(cost_out)
        disp_out = self.DispProcessor(inputs)

        return {'disp_pred': disp_out[-1],
                'train_preds': disp_out}

    def get_loss(self, model_preds, input_data):
        disp_gt = input_data["disp"]  # [bz, h, w]
        mask = (disp_gt < self.maxdisp) & (disp_gt > 0)  # [bz, h, w]
        dilated_bump_mask = input_data['dilated_bump_mask']

        weights = [0.5, 0.7, 1.0]

        loss = 0.0
        for model_pred, weight in zip(model_preds['train_preds'], weights):
            loss += weight * F.smooth_l1_loss(model_pred[mask & dilated_bump_mask], disp_gt[mask & dilated_bump_mask], reduction='mean')

        loss_info = {'scalar/train/loss_disp': loss.item()}

        return loss, loss_info
