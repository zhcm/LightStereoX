import torch
import torch.nn as nn
import torch.nn.functional as F
from stereo.modeling.common.basic_block_2d import BasicConv2d
from stereo.modeling.common.basic_block_3d import BasicConv3d
from .igev_utils.hourglass import Hourglass
from .igev_utils.backbone import Feature
from .igev_utils.igev_blocks import Conv2xUp, FeatureAtt, context_upsample, build_gwc_volume, disparity_regression
from .igev_utils.igev_blocks import build_concat_volume


class IGEVStereo(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.max_disp = cfgs.MAX_DISP
        self.num_groups = cfgs.get('NUM_GROUPS', 8)
        self.use_concat_volume = cfgs.get('USE_CONCAT_VOLUME', False)
        self.use_gwc_volume = cfgs.get('USE_GWC_VOLUME', True)
        self.concat_feature_channel = 12
        # backbobe
        self.feature = Feature()
        # get image feature
        self.stem_2 = nn.Sequential(
            BasicConv2d(3, 32,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.LeakyReLU,
                        kernel_size=3, stride=2, padding=1),
            BasicConv2d(32, 32,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.ReLU,
                        kernel_size=3, stride=1, padding=1)
        )
        self.stem_4 = nn.Sequential(
            BasicConv2d(32, 48,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.LeakyReLU,
                        kernel_size=3, stride=2, padding=1),
            BasicConv2d(48, 48,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.ReLU,
                        kernel_size=3, stride=1, padding=1),
        )
        # disp refine
        self.spx = nn.Sequential(nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1), )
        self.spx_2 = Conv2xUp(24, 32, norm_layer=nn.InstanceNorm2d, concat=True)
        self.spx_4 = nn.Sequential(
            BasicConv2d(96, 24,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.LeakyReLU,
                        kernel_size=3, stride=1, padding=1),
            BasicConv2d(24, 24,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.ReLU,
                        kernel_size=3, stride=1, padding=1))
        # conv for gwc volume
        self.conv = BasicConv2d(96, 96,
                                norm_layer=nn.InstanceNorm2d, act_layer=nn.LeakyReLU,
                                kernel_size=3, stride=1, padding=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)

        volume_channel = 0
        if self.use_gwc_volume:
            volume_channel += self.num_groups
        if self.use_concat_volume:
            volume_channel += self.concat_feature_channel * 2
        # aggregation
        self.corr_stem = BasicConv3d(volume_channel, volume_channel,
                                     norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU,
                                     kernel_size=3, stride=1, padding=1)
        self.corr_feature_att = FeatureAtt(volume_channel, 96)
        self.cost_agg = Hourglass(volume_channel)
        # cost
        self.classifier = nn.Conv3d(volume_channel, 1, 3, 1, 1, bias=False)

        if self.use_concat_volume:
            self.concat_conv = nn.Sequential(BasicConv2d(96, 32,
                                                         norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU,
                                                         kernel_size=3, stride=1, padding=1),
                                             nn.Conv2d(32, self.concat_feature_channel,
                                                       kernel_size=1, padding=0, stride=1, bias=False))

    def forward(self, data):
        image1 = data['left']
        image2 = data['right']
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()  # [bz, 3, H, W]

        # list: [bz, 48, H/4, W/4] [bz, 64, H/8, W/8] [bz, 192, H/16, W/16] [bz, 160, H/32, W/32]
        features_left = self.feature(image1)
        features_right = self.feature(image2)

        stem_2x = self.stem_2(image1)  # [bz, 32, H/2, W/2]
        stem_4x = self.stem_4(stem_2x)  # [bz, 48, H/4, W/4]
        stem_2y = self.stem_2(image2)  # [bz, 32, H/2, W/2]
        stem_4y = self.stem_4(stem_2y)  # [bz, 48, H/4, W/4]
        features_left[0] = torch.cat((features_left[0], stem_4x), 1)  # [bz, 96, H/4, W/4]
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)

        match_left = self.desc(self.conv(features_left[0]))  # [bz, 96, H/4, W/4]
        match_right = self.desc(self.conv(features_right[0]))  # [bz, 96, H/4, W/4]

        all_volume = []
        if self.use_gwc_volume:
            # [bz, num_group, max_disp/4, H/4, W/4]
            gwc_volume = build_gwc_volume(match_left, match_right, self.max_disp // 4, self.num_groups)
            all_volume.append(gwc_volume)

        if self.use_concat_volume:
            concat_feature_left = self.concat_conv(match_left)
            concat_feature_right = self.concat_conv(match_right)
            concat_volume = build_concat_volume(concat_feature_left, concat_feature_right, self.max_disp // 4)
            all_volume.append(concat_volume)

        cost_volume = torch.cat(all_volume, dim=1)
        cost_volume = self.corr_stem(cost_volume)  # [bz, num_group, max_disp/4, H/4, W/4]
        cost_volume = self.corr_feature_att(cost_volume, features_left[0])  # [bz, num_group, max_disp/4, H/4, W/4]
        geo_encoding_volume = self.cost_agg(cost_volume, features_left)  # [bz, num_group, max_disp/4, H/4, W/4]

        prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)  # [bz, max_disp/4, H/4, W/4]
        init_disp = disparity_regression(prob, self.max_disp // 4)  # [bz, 1, H/4, W/4]

        xspx = self.spx_4(features_left[0])  # [bz, 24, H/4, W/4]
        xspx = self.spx_2(xspx, stem_2x)  # [bz, 24, H/2, W/2]
        spx_pred = self.spx(xspx)  # [bz, 9, H, W]
        spx_pred = F.softmax(spx_pred, 1)  # [bz, 9, H, W]

        init_disp = context_upsample(init_disp * 4., spx_pred.float()).unsqueeze(1)  # # [bz, 1, H, W]
        return {'disp_pred': init_disp}

    def get_loss(self, model_pred, input_data):
        disp_init_pred = model_pred['disp_pred']
        disp_gt = input_data["disp"]
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)
        valid = mask.float()

        disp_gt = disp_gt.unsqueeze(1)
        mag = torch.sum(disp_gt ** 2, dim=1).sqrt()
        valid = ((valid >= 0.5) & (mag < self.max_disp)).unsqueeze(1)
        assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
        assert not torch.isinf(disp_gt[valid.bool()]).any()

        disp_loss = 1.0 * F.smooth_l1_loss(disp_init_pred[valid.bool()], disp_gt[valid.bool()], size_average=True)

        loss_info = {'scalar/train/loss_disp': disp_loss.item()}
        return disp_loss, loss_info
