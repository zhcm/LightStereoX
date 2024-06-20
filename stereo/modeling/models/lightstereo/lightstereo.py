import torch
import torch.nn as nn
import torch.nn.functional as F
from stereo.modeling.common.basic_block_2d import BasicConv2d
from stereo.modeling.common.cost_volume import build_gwc_volume, correlation_volume
from stereo.modeling.common.disp_regression import disparity_regression, topkpool_regression
from stereo.modeling.common.refinement import context_upsample
from .backbone import DeConv2x
from .backbone import Backbone
# from .v3_backbone import Backbone
# from .repvit_backbone import Backbone
# from .ultralight_vm_unet import UltraLight_VM_UNet as Backbone
# from .starnet_backbone import  Backbone
# from .aggregation import Aggregation
# from .aggregation_new import Aggregation
from .aggregation_multicost import Aggregation
from .msnet_blocks import Hourglass2D
from .ultralight_vm_unet_agg import UltraLight_VM_UNet
from .ultralight_vm_unet_3dagg import UltraLight_VM_UNet as UltraLight_VM_UNet_3D


class LightStereo(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.max_disp = cfgs.MAX_DISP

        # backbobe
        self.backbone = Backbone()

        # conv for gwc volume
        self.pre_cost_conv = nn.Sequential(
            BasicConv2d(96, 96,
                        kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU),
            nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)
        )

        # aggregation
        self.cost_agg = Aggregation(in_channels=8, gce=True)
        # self.cost_agg = Hourglass2D(in_channels=48)
        # self.cost_agg = UltraLight_VM_UNet(input_channels=48)
        # self.cost_agg = UltraLight_VM_UNet_3D(input_channels=8)

        # disp refine
        self.spx = nn.Sequential(nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1), )
        self.spx_2 = DeConv2x(24, 32)
        self.spx_4 = nn.Sequential(
            BasicConv2d(96, 24, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(24, 24, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU)
        )

    def forward(self, data):
        image1 = data['left']
        image2 = data['right']

        # list: [96, 64, 192, 160]  [bz, 32, H/2, W/2]
        features_left, stem_2x_left = self.backbone(image1)
        features_right, _ = self.backbone(image2)

        match_left = self.pre_cost_conv(features_left[0])  # [bz, 96, H/4, W/4]
        match_right = self.pre_cost_conv(features_right[0])  # [bz, 96, H/4, W/4]
        gwc_volume = build_gwc_volume(match_left, match_right, self.max_disp // 4, 8)
        gwc_volume1 = build_gwc_volume(features_left[1], features_right[1], self.max_disp // 8, 16)
        gwc_volume2 = build_gwc_volume(features_left[2], features_right[2], self.max_disp // 16, 32)
        gwc_volume3 = build_gwc_volume(features_left[3], features_right[3], self.max_disp // 32, 40)
        v_features = [gwc_volume1, gwc_volume2, gwc_volume3]
        geo_encoding_volume = self.cost_agg(v_features, gwc_volume)  # [bz, num_group, max_disp/4, H/4, W/4]

        # gwc_volume = correlation_volume(match_left, match_right, self.max_disp // 4)
        # geo_encoding_volume = self.cost_agg(gwc_volume)

        prob = F.softmax(geo_encoding_volume.squeeze(1), dim=1)  # [bz, max_disp/4, H/4, W/4]
        init_disp = disparity_regression(prob, self.max_disp // 4)  # [bz, 1, H/4, W/4]

        xspx = self.spx_4(features_left[0])  # [bz, 24, H/4, W/4]
        xspx = self.spx_2(xspx, stem_2x_left)  # [bz, 64, H/2, W/2]
        spx_pred = self.spx(xspx)  # [bz, 9, H, W]
        spx_pred = F.softmax(spx_pred, 1)  # [bz, 9, H, W]

        init_disp = context_upsample(init_disp * 4., spx_pred.float()).unsqueeze(1)  # # [bz, 1, H, W]
        return {'disp_pred': init_disp}

    def get_loss(self, model_pred, input_data):
        disp_gt = input_data["disp"]  # [bz, h, w]
        disp_gt = disp_gt.unsqueeze(1)  # [bz, 1, h, w]
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)  # [bz, 1, h, w]

        disp_pred = model_pred['disp_pred']
        loss = 1.0 * F.smooth_l1_loss(disp_pred[mask], disp_gt[mask], reduction='mean')

        loss_info = {'scalar/train/loss_disp': loss.item()}

        return loss, loss_info
