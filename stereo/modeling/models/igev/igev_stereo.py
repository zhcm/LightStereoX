import torch
import torch.nn as nn
import torch.nn.functional as F
from stereo.modeling.common.basic_block_2d import BasicConv2d
from stereo.modeling.common.basic_block_3d import BasicConv3d
from stereo.modeling.common.cost_volume import build_gwc_volume
from stereo.modeling.common.disp_regression import disparity_regression
from .igev_utils.hourglass import Hourglass
from .igev_utils.backbone import Feature
from .igev_utils.igev_blocks import Conv2xUp, FeatureAtt, context_upsample
from .igev_utils.gru_blocks import MultiBasicEncoder, CombinedGeoEncodingVolume, BasicMultiUpdateBlock


class IGEVStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.max_disp = args.MAX_DISP
        self.num_groups = args.get('NUM_GROUPS', 8)

        self.n_gru_layers = args.N_GRU_LAYERS
        self.corr_radius = args.CORR_RADIUS
        self.corr_levels = args.CORR_LEVELS
        self.slow_fast_gru = args.SLOW_FAST_GRU
        context_dims = args.HIDDEN_DIMS

        volume_channel = self.num_groups

        self.cnet = MultiBasicEncoder(output_dim=[context_dims, context_dims],
                                      norm_fn="batch",
                                      downsample=args.N_DOWNSAMPLE)
        self.update_block = BasicMultiUpdateBlock(n_gru_layers=self.n_gru_layers,
                                                  corr_levels=self.corr_levels,
                                                  corr_radius=self.corr_radius,
                                                  volume_channel=volume_channel,
                                                  hidden_dims=context_dims)

        self.context_zqr_convs = nn.ModuleList(
            [nn.Conv2d(context_dims[i], context_dims[i] * 3, 3, padding=3 // 2) for i in
             range(self.n_gru_layers)])

        # backbobe
        self.feature = Feature()
        backbone_channels = [48, 64, 192, 160]
        backbone_channels[0] = backbone_channels[0] + 48

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
            BasicConv2d(backbone_channels[0], 24,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.LeakyReLU,
                        kernel_size=3, stride=1, padding=1),
            BasicConv2d(24, 24,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.ReLU,
                        kernel_size=3, stride=1, padding=1))

        # for gru
        self.spx_2_gru = Conv2xUp(32, 32, norm_layer=nn.BatchNorm2d)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1), )

        # conv for gwc volume
        self.conv = BasicConv2d(backbone_channels[0], backbone_channels[0],
                                norm_layer=nn.InstanceNorm2d, act_layer=nn.LeakyReLU,
                                kernel_size=3, stride=1, padding=1)
        self.desc = nn.Conv2d(backbone_channels[0], backbone_channels[0], kernel_size=1, padding=0, stride=1)

        # aggregation
        self.corr_stem = BasicConv3d(volume_channel, volume_channel,
                                     norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU,
                                     kernel_size=3, stride=1, padding=1)
        self.corr_feature_att = FeatureAtt(volume_channel, backbone_channels[0])
        self.cost_agg = Hourglass(volume_channel, backbone_channels)

        # cost
        self.classifier = nn.Conv3d(volume_channel, 1, 3, 1, 1, bias=False)

    def upsample_disp(self, disp, mask_feat_4, stem_2x):
        xspx = self.spx_2_gru(mask_feat_4, stem_2x)
        spx_pred = self.spx_gru(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        up_disp = context_upsample(disp * 4., spx_pred).unsqueeze(1)
        return up_disp

    def forward(self, data):
        image1 = data['left']
        image2 = data['right']
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

        # list: [bz, 48, H/4, W/4] [bz, 64, H/8, W/8] [bz, 192, H/16, W/16] [bz, 160, H/32, W/32]
        features_left = self.feature(image1)
        features_right = self.feature(image2)

        stem_2x = self.stem_2(image1)  # [bz, 32, H/2, W/2]
        stem_4x = self.stem_4(stem_2x)  # [bz, 48, H/4, W/4]
        stem_2y = self.stem_2(image2)  # [bz, 32, H/2, W/2]
        stem_4y = self.stem_4(stem_2y)  # [bz, 48, H/4, W/4]
        features_left[0] = torch.cat((features_left[0], stem_4x), 1)  # [bz, 96, H/4, W/4]
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)  # [bz, 96, H/4, W/4]

        match_left = self.desc(self.conv(features_left[0]))  # [bz, 96, H/4, W/4]
        match_right = self.desc(self.conv(features_right[0]))  # [bz, 96, H/4, W/4]

        # [bz, num_group, max_disp/4, H/4, W/4]
        gwc_volume = build_gwc_volume(match_left, match_right, self.max_disp // 4, self.num_groups)  # [bz, num_group, max_disp/4, H/4, W/4]
        cost_volume = self.corr_stem(gwc_volume)
        cost_volume = self.corr_feature_att(cost_volume, features_left[0])
        geo_encoding_volume = self.cost_agg(cost_volume, features_left)

        prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)
        init_disp = disparity_regression(prob, self.max_disp // 4)

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

        iters = self.args.TRAIN_ITERS if self.training else self.args.EVAL_ITERS
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

        result = {'disp_preds': disp_preds, 'disp_pred': disp_preds[-1]}
        if self.training:
            xspx = self.spx_4(features_left[0])
            xspx = self.spx_2(xspx, stem_2x)
            spx_pred = self.spx(xspx)
            spx_pred = F.softmax(spx_pred, 1)
            init_disp = context_upsample(init_disp * 4., spx_pred.float()).unsqueeze(1)
            result['init_disp'] = init_disp

        return result

    def get_loss(self, model_pred, input_data):
        disp_init_pred = model_pred['init_disp']
        disp_gt = input_data["disp"]
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)
        valid = mask.float()

        disp_gt = disp_gt.unsqueeze(1)
        mag = torch.sum(disp_gt ** 2, dim=1).sqrt()
        valid = ((valid >= 0.5) & (mag < self.max_disp)).unsqueeze(1)
        assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
        assert not torch.isinf(disp_gt[valid.bool()]).any()
        disp_loss = 1.0 * F.smooth_l1_loss(disp_init_pred[valid.bool()], disp_gt[valid.bool()], reduction='mean')

        # gru loss
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

        loss_info = {'scalar/train/loss_disp': disp_loss.item()}
        return disp_loss, loss_info
