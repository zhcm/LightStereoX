import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from stereo.modeling.common.basic_block_2d import BasicConv2d, BasicDeconv2d
from stereo.modeling.common.cost_volume import correlation_volume, build_gwc_volume
from stereo.modeling.common.disp_regression import disparity_regression
from stereo.modeling.common.refinement import context_upsample, BasicBlock

from .backbone import Backbone, FPNLayer
# from .backbone_repvit import Backbone, FPNLayer
# from .backbone_starnet import Backbone, FPNLayer

from .aggregation import Aggregation
# from .aggregation_new import Aggregation
# from .aggregation_vit import Aggregation
# from .aggregation_vit_new import Aggregation


class Refinement(nn.Module):
    def __init__(self):
        super(Refinement, self).__init__()

        # Original StereoNet: left, disp
        self.conv = BasicConv2d(4, 32, kernel_size=3, stride=1, padding=1,
                                norm_layer=nn.BatchNorm2d,
                                act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True))

        self.dilation_list = [2]
        self.dilated_blocks = nn.ModuleList()

        for dilation in self.dilation_list:
            self.dilated_blocks.append(BasicBlock(32, 32, stride=1,
                                                  padding=dilation, dilation=dilation))

        self.dilated_blocks = nn.Sequential(*self.dilated_blocks)

        self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, low_disp, left_img):
        """Upsample low resolution disparity prediction to
        corresponding resolution as image size
        Args:
            low_disp: [B, 1, H, W]
            left_img: [B, 3, H, W]
        """
        scale_factor = left_img.size(-1) / low_disp.size(-1)
        if scale_factor == 1.0:
            disp = low_disp
        else:
            disp = F.interpolate(low_disp, size=left_img.size()[-2:], mode='bilinear', align_corners=False)
            disp = disp * scale_factor  # scale correspondingly

        concat = torch.cat((disp, left_img), dim=1)  # [B, 4, H, W]
        out = self.conv(concat)
        out = self.dilated_blocks(out)
        residual_disp = self.final_conv(out)

        disp = F.relu(disp + residual_disp, inplace=True)  # [B, 1, H, W]

        return disp


class LightStereo(nn.Module):
    def __init__(self, max_disp, aggregation_blocks, expanse_ratio, left_att=True):
        super().__init__()
        self.max_disp = max_disp
        self.left_att = left_att

        # backbobe
        self.backbone = Backbone()

        # aggregation
        self.cost_agg = Aggregation(in_channels=48,
                                    left_att=self.left_att,
                                    blocks=aggregation_blocks,
                                    expanse_ratio=expanse_ratio)

        # disp refine
        self.refine_1 = nn.Sequential(
            BasicConv2d(24, 24, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(24, 24, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.ReLU))

        self.stem_2 = nn.Sequential(
            BasicConv2d(3, 16, kernel_size=3, stride=2, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(16, 16, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU))
        self.refine_2 = FPNLayer(24, 16)
        # self.refine_2 = BasicDeconv2d(24, 24, kernel_size=4, stride=2, padding=1)

        self.refine_3 = BasicDeconv2d(16, 9, kernel_size=4, stride=2, padding=1)

        # self.refinemet = Refinement()

        # for module in self.modules():
        #     for name, child in module.named_children():
        #         if isinstance(child, nn.BatchNorm2d):
        #             gn = nn.InstanceNorm2d(child.num_features)
        #             setattr(module, name, gn)

    def forward(self, data):
        image1 = data['left']
        image2 = data['right']

        features_left = self.backbone(image1)
        features_right = self.backbone(image2)
        # features_left, features_right = self.backbone(image1, image2)

        gwc_volume = correlation_volume(features_left[0], features_right[0], self.max_disp // 4)
        encoding_volume = self.cost_agg(gwc_volume, features_left)  # [bz, 1, max_disp/4, H/4, W/4]
        # gwc_volume = build_gwc_volume(features_left[0], features_right[0], self.max_disp // 4, num_groups=8)
        # encoding_volume = self.cost_agg(gwc_volume, features_left)

        prob = F.softmax(encoding_volume[0].squeeze(1), dim=1)  # [bz, max_disp/4, H/4, W/4]
        init_disp = disparity_regression(prob, self.max_disp // 4)  # [bz, 1, H/4, W/4]

        xspx = self.refine_1(features_left[0])
        xspx = self.refine_2(xspx, self.stem_2(image1))
        xspx = self.refine_3(xspx)
        spx_pred = F.softmax(xspx, 1)  # [bz, 9, H, W]
        disp_pred = context_upsample(init_disp * 4., spx_pred.float()).unsqueeze(1)  # # [bz, 1, H, W]

        # disp_pred = self.refinemet(disp_pred, image1)
        result = {'disp_pred': disp_pred}

        # if self.training:
        #     disp_4 = F.interpolate(init_disp, image1.shape[2:], mode='bilinear', align_corners=False)
        #     disp_4 *= 4
        #     result['disp_4'] = disp_4

            # prob = F.softmax(encoding_volume[1].squeeze(1), dim=1)
            # init_disp = disparity_regression(prob, self.max_disp // 4)
            # visual_disp_4 = F.interpolate(init_disp, image1.shape[2:], mode='bilinear', align_corners=False)
            # visual_disp_4 *= 4
            # result['visual_disp_4'] = visual_disp_4
            #
            # prob = F.softmax(encoding_volume[2].squeeze(1), dim=1)
            # init_disp = disparity_regression(prob, self.max_disp // 8)
            # visual_disp_8_0 = F.interpolate(init_disp, image1.shape[2:], mode='bilinear', align_corners=False)
            # visual_disp_8_0 *= 8
            # result['visual_disp_8_0'] = visual_disp_8_0
            #
            # prob = F.softmax(encoding_volume[3].squeeze(1), dim=1)
            # init_disp = disparity_regression(prob, self.max_disp // 8)
            # visual_disp_8_1 = F.interpolate(init_disp, image1.shape[2:], mode='bilinear', align_corners=False)
            # visual_disp_8_1 *= 8
            # result['visual_disp_8_1'] = visual_disp_8_1
            #
            # prob = F.softmax(encoding_volume[4].squeeze(1), dim=1)
            # init_disp = disparity_regression(prob, self.max_disp // 16)
            # visual_disp_16_0 = F.interpolate(init_disp, image1.shape[2:], mode='bilinear', align_corners=False)
            # visual_disp_16_0 *= 16
            # result['visual_disp_16_0'] = visual_disp_16_0
            #
            # prob = F.softmax(encoding_volume[5].squeeze(1), dim=1)
            # init_disp = disparity_regression(prob, self.max_disp // 16)
            # visual_disp_16_1 = F.interpolate(init_disp, image1.shape[2:], mode='bilinear', align_corners=False)
            # visual_disp_16_1 *= 16
            # result['visual_disp_16_1'] = visual_disp_16_1

        return result

    def get_loss(self, model_pred, input_data):
        disp_gt = input_data["disp"]  # [bz, h, w]
        disp_gt = disp_gt.unsqueeze(1)  # [bz, 1, h, w]
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)  # [bz, 1, h, w]

        disp_pred = model_pred['disp_pred']
        loss = 1.0 * F.smooth_l1_loss(disp_pred[mask], disp_gt[mask], reduction='mean')

        # disp_4 = model_pred['disp_4']
        # loss += 0.3 * F.smooth_l1_loss(disp_4[mask], disp_gt[mask], reduction='mean')

        # visual_disp_4 = model_pred['visual_disp_4']
        # loss += 0.1 * F.smooth_l1_loss(visual_disp_4[mask], disp_gt[mask], reduction='mean')
        # visual_disp_8_0 = model_pred['visual_disp_8_0']
        # loss += 0.1 * F.smooth_l1_loss(visual_disp_8_0[mask], disp_gt[mask], reduction='mean')
        # visual_disp_8_1 = model_pred['visual_disp_8_1']
        # loss += 0.1 * F.smooth_l1_loss(visual_disp_8_1[mask], disp_gt[mask], reduction='mean')
        # visual_disp_16_0 = model_pred['visual_disp_16_0']
        # loss += 0.1 * F.smooth_l1_loss(visual_disp_16_0[mask], disp_gt[mask], reduction='mean')
        # visual_disp_16_1 = model_pred['visual_disp_16_1']
        # loss += 0.1 * F.smooth_l1_loss(visual_disp_16_1[mask], disp_gt[mask], reduction='mean')

        loss_info = {'scalar/train/loss_disp': loss.item()}

        return loss, loss_info
