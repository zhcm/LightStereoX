# @Time    : 2024/4/2 10:53
# @Author  : zhangchenming
import torch
import torch.nn as nn
import torch.nn.functional as F
from .coex_backbone import CoExBackbone
from .coex_cost_processor import CoExCostProcessor
from .coex_disp_processor import CoExDispProcessor
from functools import partial


class CoEx(nn.Module):
    def __init__(self):
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

    def forward(self, inputs):
        """Forward the network."""
        backbone_out = self.Backbone(inputs)
        inputs.update(backbone_out)
        cost_out = self.CostProcessor(inputs)
        inputs.update(cost_out)
        disp_out = self.DispProcessor(inputs)

        if self.training:
            pred_height = self.height_head(torch.cat([inputs["left"], disp_out['disp_ests'][0].unsqueeze(1)], dim=1))
            return {'disp_preds': disp_out['disp_ests'],
                    'disp_pred': disp_out['disp_ests'][0],
                    'pred_height': pred_height.squeeze(1)}
        else:
            pred_height = self.height_head(torch.cat([inputs["left"], disp_out['inference_disp']['disp_est'].unsqueeze(1)], dim=1))
            return {'disp_pred': disp_out['inference_disp']['disp_est'],
                    'pred_height': pred_height.squeeze(1)}

    def get_loss(self, model_preds, input_data):
        disp_gt = input_data["disp"]  # [bz, h, w]
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)  # [bz, h, w]

        weights = [1.0, 0.3]

        loss = 0.0
        for disp_est, weight in zip(model_preds['disp_preds'], weights):
            loss += weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True)

        loss = loss * 0.77
        loss_info = {'scalar/train/loss_disp': loss.item()}

        # height_loss = F.smooth_l1_loss(model_preds['pred_height'], input_data['height_map'], size_average=True)
        bump_mask = input_data['crop_area']
        height_loss = F.smooth_l1_loss(model_preds['pred_height'][bump_mask], input_data['bump_height_map'][bump_mask], size_average=True)

        loss_info['scalar/train/loss_height'] = height_loss.item()
        return loss + height_loss, loss_info


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


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, padding=0, dilation=1):
        super().__init__()
        self.block_1 = BasicConv2d(in_channels=channels, out_channels=channels,
                                   norm_layer=nn.BatchNorm2d, act_layer=partial(nn.LeakyReLU, negative_slope=0.2),
                                   kernel_size=kernel_size, padding=padding, dilation=dilation)

        self.block_2 = BasicConv2d(in_channels=channels, out_channels=channels,
                                   norm_layer=nn.BatchNorm2d, act_layer=None,
                                   kernel_size=kernel_size, padding=padding, dilation=dilation)

        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        res = self.block_1(x)
        res = self.block_2(res)
        out = res + x
        out = self.activation(out)
        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False,
                 norm_layer=None, act_layer=None, **kwargs):
        super(BasicConv2d, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, **kwargs)]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if act_layer is not None:
            layers.append(act_layer())

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x
