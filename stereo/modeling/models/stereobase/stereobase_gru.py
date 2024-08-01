import torch
import torch.nn as nn
import torch.nn.functional as F
from stereo.modeling.models.lightstereo.basic_block_2d import BasicConv2d
from stereo.modeling.common.cost_volume import build_concat_volume, build_gwc_volume
from stereo.modeling.common.disp_regression import disparity_regression
from stereo.modeling.common.refinement import context_upsample

from .neck import Neck, FPNLayer
from .hourglass import Hourglass
from .gru_blocks import MultiBasicEncoder, CombinedGeoEncodingVolume, BasicMultiUpdateBlock


class StereoBase(nn.Module):
    def __init__(self, backbone, max_disp, num_groups, concat_channels,
                 context_dims, n_downsample, n_gru_layers, corr_radius, corr_levels, slow_fast_gru,
                 train_iters, eval_iters):
        super().__init__()
        self.max_disp = max_disp
        self.num_groups = num_groups
        self.concat_channels = concat_channels
        self.n_gru_layers = n_gru_layers
        self.corr_radius = corr_radius
        self.corr_levels = corr_levels
        self.slow_fast_gru = slow_fast_gru
        self.train_iters = train_iters
        self.eval_iters = eval_iters

        self.backbone = backbone
        backbone_dims = self.backbone.out_dims
        channels = [backbone_dims['scale2'], backbone_dims['scale3'], backbone_dims['scale4'], backbone_dims['scale5']]
        self.neck = Neck(channels)

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

        channels = [backbone_dims['scale2'] * 2 + 48, backbone_dims['scale3'] * 2, backbone_dims['scale4'] * 2, backbone_dims['scale5']]

        self.conv = BasicConv2d(channels[0], channels[0],
                                norm_layer=nn.InstanceNorm2d, act_layer=nn.LeakyReLU,
                                kernel_size=3, stride=1, padding=1)
        self.desc = nn.Conv2d(channels[0], channels[0], kernel_size=1, padding=0, stride=1)

        self.concat_conv = nn.Sequential(
            BasicConv2d(channels[0], 32, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU,
                        kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, self.concat_channels, kernel_size=1, padding=0, stride=1, bias=False)
        )

        volume_channel = self.num_groups + self.concat_channels * 2

        self.cost_agg = Hourglass(volume_channel, channels)
        self.classifier = nn.Conv3d(volume_channel, 1, 3, 1, 1, bias=False)

        self.cnet = MultiBasicEncoder(output_dim=[context_dims, context_dims],
                                      norm_fn="batch",
                                      downsample=n_downsample)
        self.update_block = BasicMultiUpdateBlock(n_gru_layers=self.n_gru_layers,
                                                  corr_levels=self.corr_levels,
                                                  corr_radius=self.corr_radius,
                                                  volume_channel=volume_channel,
                                                  hidden_dims=context_dims)
        self.context_zqr_convs = nn.ModuleList(
            [nn.Conv2d(context_dims[i], context_dims[i] * 3, 3, padding=3 // 2) for i in
             range(self.n_gru_layers)])

        self.spx_2_gru = FPNLayer(32, 32, norm_layer=nn.BatchNorm2d)
        self.spx_gru = nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1)

        # disp refine
        self.spx = nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1)
        self.spx_2 = FPNLayer(24, 32, norm_layer=nn.InstanceNorm2d)
        self.spx_4 = nn.Sequential(
            BasicConv2d(channels[0], 24,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.LeakyReLU,
                        kernel_size=3, stride=1, padding=1),
            BasicConv2d(24, 24,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.ReLU,
                        kernel_size=3, stride=1, padding=1))

    def upsample_disp(self, disp, mask_feat_4, stem_2x):
        xspx = self.spx_2_gru(mask_feat_4, stem_2x)
        spx_pred = self.spx_gru(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        up_disp = context_upsample(disp * 4., spx_pred).unsqueeze(1)
        return up_disp

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

        stem_2x = self.stem_2(image1)  # [bz, 32, H/2, W/2]
        stem_4x = self.stem_4(stem_2x)  # [bz, 48, H/4, W/4]
        stem_2y = self.stem_2(image2)  # [bz, 32, H/2, W/2]
        stem_4y = self.stem_4(stem_2y)  # [bz, 48, H/4, W/4]

        features_left[0] = torch.cat((features_left[0], stem_4x), 1)  # [bz, 96, H/4, W/4]
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)

        match_left = self.desc(self.conv(features_left[0]))  # [bz, 96, H/4, W/4]
        match_right = self.desc(self.conv(features_right[0]))  # [bz, 96, H/4, W/4]

        gwc_volume = build_gwc_volume(match_left, match_right, self.max_disp // 4, self.num_groups)  # [bz, num_group, max_disp/4, H/4, W/4]

        concat_feature_left = self.concat_conv(match_left)
        concat_feature_right = self.concat_conv(match_right)
        concat_volume = build_concat_volume(concat_feature_left, concat_feature_right, self.max_disp // 4)  # [bz, concat_c * 2, max_disp/4, H/4, W/4]

        cost_volume = torch.cat([gwc_volume, concat_volume], dim=1)
        geo_encoding_volume = self.cost_agg(cost_volume, features_left)  # [bz, channel, max_disp/4, H/4, W/4]

        prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)  # [bz, max_disp/4, H/4, W/4]
        init_disp = disparity_regression(prob, self.max_disp // 4)  # [bz, 1, H/4, W/4]

        # gru
        cnet_list = self.cnet(image1, num_layers=self.n_gru_layers)
        net_list = [torch.tanh(x[0]) for x in cnet_list]
        inp_list = [torch.relu(x[1]) for x in cnet_list]
        inp_list = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1)) for i, conv in
                    zip(inp_list, self.context_zqr_convs)]
        geo_fn = CombinedGeoEncodingVolume(match_left.float(),
                                           match_right.float(),
                                           geo_encoding_volume.float(),
                                           radius=self.corr_radius,
                                           num_levels=self.corr_levels)
        b, c, h, w = match_left.shape
        coords = torch.arange(w).float().to(match_left.device).reshape(1, 1, w, 1).repeat(b, h, 1, 1)  # [1, 1, W/4, 1] -> [bz, H/4, W/4, 1]
        disp = init_disp
        disp_preds = []

        iters = self.train_iters if self.training else self.eval_iters
        for itr in range(iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords)  # [bz, (channel+1)*(2r+1)*corr_levels, H/4, W/4]
            if self.n_gru_layers == 3 and self.slow_fast_gru:  # Update low-res ConvGRU
                net_list = self.update_block(net_list, inp_list,
                                             iter16=True,
                                             iter08=False,
                                             iter04=False,
                                             update=False)
            if self.n_gru_layers >= 2 and self.slow_fast_gru:  # Update low-res ConvGRU and mid-res ConvGRU
                net_list = self.update_block(net_list, inp_list,
                                             iter16=self.n_gru_layers == 3,
                                             iter08=True,
                                             iter04=False,
                                             update=False)
            net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp,
                                                                  iter16=self.n_gru_layers == 3,
                                                                  iter08=self.n_gru_layers >= 2)
            disp = disp + delta_disp
            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)
            disp_preds.append(disp_up)

        xspx = self.spx_4(features_left[0])  # [bz, 24, H/4, W/4]
        xspx = self.spx_2(xspx, stem_2x)  # [bz, 24, H/2, W/2]
        spx_pred = self.spx(xspx)  # [bz, 9, H, W]
        spx_pred = F.softmax(spx_pred, 1)  # [bz, 9, H, W]
        init_disp = context_upsample(init_disp * 4., spx_pred.float()).unsqueeze(1)  # [bz, 1, H, W]

        return {'init_disp': init_disp,
                'disp_preds': disp_preds,
                'disp_pred': disp_preds[-1]}

    def get_loss(self, model_pred, input_data):
        disp_gt = input_data["disp"]  # [bz, h, w]
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)  # [bz, h, w]
        valid = mask.float()  # [bz, h, w]

        disp_gt = disp_gt.unsqueeze(1)  # [bz, 1, h, w]
        mag = torch.sum(disp_gt ** 2, dim=1).sqrt()  # [bz, h, w]
        valid = ((valid >= 0.5) & (mag < self.max_disp)).unsqueeze(1)  # [bz, 1, h, w]
        assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
        assert not torch.isinf(disp_gt[valid.bool()]).any()

        disp_init_pred = model_pred['init_disp']
        disp_loss = 1.0 * F.smooth_l1_loss(disp_init_pred[valid.bool()], disp_gt[valid.bool()], reduction='mean')

        loss_gamma = 0.9
        disp_preds = model_pred['disp_preds']
        n_predictions = len(disp_preds)
        assert n_predictions >= 1
        for i in range(n_predictions):
            adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
            i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
            i_loss = (disp_preds[i] - disp_gt).abs()
            assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, disp_preds[i].shape]
            disp_loss += i_weight * i_loss[valid.bool()].mean()

        tb_info = {'scalar/train/loss_disp': disp_loss.item()}

        return disp_loss, tb_info
