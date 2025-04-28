import torch
import torch.nn as nn
import torch.nn.functional as F

from stereo.modeling.common.cost_volume import correlation_volume
from stereo.modeling.common.disp_regression import disparity_regression
from stereo.modeling.common.refinement import context_upsample

from stereo.modeling.models.lightstereo.basic_block_2d import BasicConv2d, BasicDeconv2d
from stereo.modeling.models.lightstereo.neck import Neck, FPNLayer
from stereo.modeling.models.lightstereo.aggregation import Aggregation

from .rbhm import Refinement


class LightStereo(nn.Module):
    def __init__(self, backbone, max_disp, aggregation_blocks, expanse_ratio, left_att=True, rbhm_pretrained=''):
        super().__init__()
        self.max_disp = max_disp
        self.left_att = left_att

        # backbobe
        self.backbone = backbone
        backbone_dims = self.backbone.out_dims
        channels = [backbone_dims['scale2'], backbone_dims['scale3'], backbone_dims['scale4'], backbone_dims['scale5']]
        self.neck = Neck(channels)

        # aggregation
        self.cost_agg = Aggregation(in_channels=48,
                                    left_att=self.left_att,
                                    blocks=aggregation_blocks,
                                    expanse_ratio=expanse_ratio,
                                    backbone_dims=channels)

        # disp refine
        self.refine_1 = nn.Sequential(
            BasicConv2d(channels[0], 24, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(24, 24, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.ReLU))

        self.stem_2 = nn.Sequential(
            BasicConv2d(3, 16, kernel_size=3, stride=2, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(16, 16, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU))
        self.refine_2 = FPNLayer(24, 16)

        self.refine_3 = BasicDeconv2d(16, 9, kernel_size=4, stride=2, padding=1)

        self.height_head = Refinement(in_channels=4)
        pretrained_state_dict = torch.load(rbhm_pretrained, map_location='cpu')
        state_dict = {}
        for key, val in pretrained_state_dict.items():
            state_dict[key.replace('height_head.', '')] = val
        self.height_head.load_state_dict(state_dict)

    def forward(self, data):
        image1 = data['left']
        image2 = data['right']
        bz, _, _, _ = image1.shape

        src_images = torch.cat([image1, image2], dim=0)
        features = self.backbone(src_images)
        n_inputs = [features['scale2'], features['scale3'], features['scale4'], features['scale5']]
        features = self.neck(n_inputs)
        features_left = [feat[:bz] for feat in features]
        features_right = [feat[bz:] for feat in features]

        cost_volume = correlation_volume(features_left[0], features_right[0], self.max_disp // 4)
        encoding_volume = self.cost_agg(cost_volume, features_left)  # [bz, 1, max_disp/4, H/4, W/4]

        prob = F.softmax(encoding_volume[0].squeeze(1), dim=1)  # [bz, max_disp/4, H/4, W/4]
        init_disp = disparity_regression(prob, self.max_disp // 4)  # [bz, 1, H/4, W/4]

        xspx = self.refine_1(features_left[0])
        xspx = self.refine_2(xspx, self.stem_2(image1))
        xspx = self.refine_3(xspx)
        spx_pred = F.softmax(xspx, 1)  # [bz, 9, H, W]
        disp_pred = context_upsample(init_disp * 4., spx_pred.float()).unsqueeze(1)  # # [bz, 1, H, W]

        pred_height = self.height_head(torch.cat([data["left"], disp_pred], dim=1))
        pred_height = torch.sigmoid(pred_height) * 10
        result = {'disp_pred': disp_pred,
                  'pred_height': pred_height.squeeze(1)}

        if self.training:
            disp_4 = F.interpolate(init_disp, image1.shape[2:], mode='bilinear', align_corners=False)
            disp_4 *= 4
            result['disp_4'] = disp_4

        return result
