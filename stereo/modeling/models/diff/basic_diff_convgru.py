import torch
import torch.nn as nn
import torch.nn.functional as F

from stereo.modeling.common.basic_block_2d import BasicConv2d
from stereo.modeling.common.basic_block_3d import BasicConv3d
from stereo.modeling.common.cost_volume import build_gwc_volume
from stereo.modeling.common.disp_regression import disparity_regression

from stereo.modeling.models.igev.igev_utils.backbone import Feature
from stereo.modeling.models.igev.igev_utils.hourglass import Hourglass
from stereo.modeling.models.igev.igev_utils.igev_blocks import Conv2xUp, FeatureAtt, context_upsample
from stereo.modeling.models.igev.igev_utils.gru_blocks import MultiBasicEncoder, CombinedGeoEncodingVolume, BasicMultiUpdateBlock

from .disp2prob import disp2prob
from .scheduling_ddim import DDIMScheduler
from .pipeline_ddim import DDIMPipeline, Stereo2DModel


class IgevNoGruDiff(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.max_disp = cfgs.MAX_DISP
        self.num_groups = cfgs.get('NUM_GROUPS', 8)
        self.use_concat_volume = cfgs.get('USE_CONCAT_VOLUME', False)
        self.use_gwc_volume = cfgs.get('USE_GWC_VOLUME', True)
        self.use_sub_volume = cfgs.get('USE_SUB_VOLUME', False)
        self.use_interlaced_volume = cfgs.get('USE_INTERLACED_VOLUME', False)
        self.concat_feature_channel = cfgs.get('CONCAT_CHANNELS', 12)
        self.interlaced_feature_channel = cfgs.get('INTERLACED_CHANNELS', 8)
        self.backbone_name = cfgs.get('BACKBONE', 'mobilenetv2_100')

        self.n_gru_layers = cfgs.N_GRU_LAYERS
        self.corr_radius = cfgs.CORR_RADIUS
        self.corr_levels = cfgs.CORR_LEVELS
        self.iters = cfgs.ITERS
        self.slow_fast_gru = cfgs.SLOW_FAST_GRU
        context_dims = cfgs.HIDDEN_DIMS

        volume_channel = 0
        if self.use_gwc_volume:
            volume_channel += self.num_groups
        if self.use_concat_volume:
            volume_channel += self.concat_feature_channel * 2
        if self.use_sub_volume:
            volume_channel += 1
        if self.use_interlaced_volume:
            volume_channel += self.interlaced_feature_channel

        self.cnet = MultiBasicEncoder(output_dim=[context_dims, context_dims],
                                      norm_fn="batch",
                                      downsample=cfgs.N_DOWNSAMPLE)
        self.update_block = BasicMultiUpdateBlock(n_gru_layers=self.n_gru_layers,
                                                  corr_levels=self.corr_levels,
                                                  corr_radius=self.corr_radius,
                                                  volume_channel=volume_channel,
                                                  hidden_dims=context_dims)
        self.context_zqr_convs = nn.ModuleList(
            [nn.Conv2d(context_dims[i], context_dims[i] * 3, 3, padding=3 // 2) for i in
             range(self.n_gru_layers)])
        self.spx_2_gru = Conv2xUp(32, 32, norm_layer=nn.BatchNorm2d)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1), )

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

        # diffusers
        self.scheduler = DDIMScheduler(num_train_timesteps=1000, clip_sample=False)
        self.diff_model = Stereo2DModel(channels_in=48, num_train_timesteps=1000)
        self.pipeline = DDIMPipeline(self.diff_model, self.scheduler)

    def get_gt_prob(self, data, size):
        disp_gt = data['disp'].unsqueeze(1)
        disp_gt = F.interpolate(disp_gt, size, mode='nearest')
        disp_gt = disp_gt / 4

        lower_bound = 0
        upper_bound = self.max_disp // 4
        mask = (disp_gt > lower_bound) & (disp_gt < upper_bound)
        disp_gt = disp_gt * mask

        disp_gt_prob = disp2prob(max_disp=self.max_disp // 4, disp_gt=disp_gt)
        return disp_gt_prob

    def forward(self, data):
        image1 = data['left']
        image2 = data['right']
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()  # [bz, 3, H, W]
        features_left = self.feature(image1)  # 1/4, 1/8, 1/16, 1/32
        features_right = self.feature(image2)
        stem_2x = self.stem_2(image1)
        stem_4x = self.stem_4(stem_2x)
        stem_2y = self.stem_2(image2)
        stem_4y = self.stem_4(stem_2y)
        features_left[0] = torch.cat((features_left[0], stem_4x), 1)  # bz, 96, h/4, w/4
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)

        match_left = self.desc(self.conv(features_left[0]))
        match_right = self.desc(self.conv(features_right[0]))
        gwc_volume = build_gwc_volume(match_left, match_right, self.max_disp // 4, 8)
        # [4, 8, 48, 64, 128] [batch, channel, disp, w, h]

        cost_volume = self.corr_stem(gwc_volume)
        cost_volume = self.corr_feature_att(cost_volume, features_left[0])
        geo_encoding_volume = self.cost_agg(cost_volume, features_left)
        prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)  # [bz, max_disp/4, H/4, W/4]

        prob = prob * 2 - 1
        diff_probs = self.pipeline(cond_feature=gwc_volume, num_inference_steps=5)
        diff_prob = (diff_probs[-1] / 2 + 0.5).clamp(0, 1)
        volume_filter = diff_prob.unsqueeze(1)
        filted_volume = gwc_volume * volume_filter

        cost_volume = self.corr_stem(filted_volume)
        cost_volume = self.corr_feature_att(cost_volume, features_left[0])
        geo_encoding_volume = self.cost_agg(cost_volume, features_left)

        new_prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)
        new_init_disp = disparity_regression(new_prob, self.max_disp // 4)

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
        disp = new_init_disp

        disp_preds = []
        for itr in range(self.iters):
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

        result = {}
        if self.training:
            init_disp = disparity_regression(prob, self.max_disp // 4)
            xspx = self.spx_4(features_left[0])
            xspx = self.spx_2(xspx, stem_2x)
            spx_pred = self.spx(xspx)
            spx_pred = F.softmax(spx_pred, 1)  # bz, 9, h, w
            init_disp = context_upsample(init_disp * 4., spx_pred.float()).unsqueeze(1)  # bz, 1, h, w
            new_init_disp = context_upsample(new_init_disp * 4., spx_pred.float()).unsqueeze(1)  # [bz, 1, H, W]
            result['init_disp'] = init_disp
            result['new_init_disp'] = new_init_disp

            disp_gt_prob = self.get_gt_prob(data, gwc_volume.shape[-2:])  # [bz, 48, 80, 184]
            disp_gt_prob = disp_gt_prob * 2 - 1
            noise = torch.randn(disp_gt_prob.shape, dtype=disp_gt_prob.dtype, device=disp_gt_prob.device)
            bz = disp_gt_prob.shape[0]
            timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bz,), device=disp_gt_prob.device).long()
            noisy_images = self.scheduler.add_noise(disp_gt_prob, noise, timesteps)
            noise_pred = self.diff_model(noisy_images, timesteps, prob)
            result['noise'] = noise
            result['noise_pred'] = noise_pred

        result['disp_preds'] = disp_preds
        result['disp_pred'] = disp_preds[-1]

        return result

    def get_loss(self, model_pred, input_data):

        disp_gt = input_data["disp"]  # [bz, h, w]
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)  # [bz, h, w]
        valid = mask.float()  # [bz, h, w]

        disp_gt = disp_gt.unsqueeze(1)  # [bz, 1, h, w]
        mag = torch.sum(disp_gt ** 2, dim=1).sqrt()  # [bz, h, w]
        valid = ((valid >= 0.5) & (mag < self.max_disp)).unsqueeze(1)  # [bz, 1, h, w]
        assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
        assert not torch.isinf(disp_gt[valid.bool()]).any()

        loss_info = {}

        # init loss
        disp_init_pred = model_pred['disp_init']
        disp_init_loss = 1.0 * F.smooth_l1_loss(disp_init_pred[valid.bool()], disp_gt[valid.bool()], reduction='mean')
        loss_info['scalar/train/disp_init'] = disp_init_loss.item()

        # new_init loss
        new_disp_init_pred = model_pred['new_init_disp']
        new_disp_init_loss = 1.0 * F.smooth_l1_loss(new_disp_init_pred[valid.bool()], disp_gt[valid.bool()], reduction='mean')
        loss_info['scalar/train/new_disp_init'] = new_disp_init_loss.item()

        # gru loss
        loss_gamma = 0.9
        disp_preds = model_pred['disp_preds']
        n_predictions = len(disp_preds)
        assert n_predictions >= 1
        disp_preds_loss = 0
        for i in range(n_predictions):
            adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
            i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
            i_loss = (disp_preds[i] - disp_gt).abs()
            disp_preds_loss += i_weight * i_loss[valid.bool()].mean()
        loss_info['scalar/train/disp_preds_loss'] = disp_preds_loss.item()

        # diff noise pred loss
        noise = model_pred['noise']
        noise_pred = model_pred['noise_pred']
        loss_mse = F.mse_loss(noise_pred, noise)
        loss_info['scalar/train/loss_mse'] = loss_mse.item()

        loss = disp_init_loss + new_disp_init_loss + disp_preds_loss + loss_mse
        return loss, loss_info
