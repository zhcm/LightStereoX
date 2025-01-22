# @Time    : 2025/1/15 18:17
# @Author  : zhangchenming
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial


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


class HeightPred(nn.Module):
    def __init__(self):
        super().__init__()
        self.height_head = Refinement(in_channels=4)

    def forward(self, inputs):
        pred_height = self.height_head(torch.cat([inputs["left"], inputs['disp'].unsqueeze(1)], dim=1))
        pred_height = torch.sigmoid(pred_height) * 10
        return {'pred_height': pred_height.squeeze(1)}

    @staticmethod
    def get_loss(model_preds, input_data):
        dilated_bump_mask = input_data['dilated_bump_mask']
        bump_mask = input_data['bump_mask']
        height_loss = F.smooth_l1_loss(model_preds['pred_height'][bump_mask], input_data['height_map'][bump_mask], size_average=True)
        loss_info = {'scalar/train/loss_height': height_loss.item()}
        return height_loss, loss_info
