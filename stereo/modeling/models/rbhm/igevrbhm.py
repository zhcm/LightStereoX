from stereo.modeling.models.igev.extractor import MultiBasicEncoder, Feature
from stereo.modeling.models.igev.geometry import Combined_Geo_Encoding_Volume
from stereo.modeling.models.igev.submodule import *
from stereo.modeling.models.igev.update import BasicMultiUpdateBlock
from stereo.modeling.models.igev.utils import Map

from stereo.modeling.models.rbhm.rbhm import Refinement


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(
            BasicConv(in_channels, in_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=1, dilation=1))

        self.conv2 = nn.Sequential(
            BasicConv(in_channels * 2, in_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 4, in_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=1, dilation=1))

        self.conv3 = nn.Sequential(
            BasicConv(in_channels * 4, in_channels * 6, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 6, in_channels * 6, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=1, dilation=1))

        self.conv3_up = BasicConv(in_channels * 6, in_channels * 4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels * 4, in_channels * 2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels * 2, 8, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(
            BasicConv(in_channels * 8, in_channels * 4, is_3d=True, kernel_size=1, padding=0, stride=1),
            BasicConv(in_channels * 4, in_channels * 4, is_3d=True, kernel_size=3, padding=1, stride=1),
            BasicConv(in_channels * 4, in_channels * 4, is_3d=True, kernel_size=3, padding=1, stride=1), )

        self.agg_1 = nn.Sequential(
            BasicConv(in_channels * 4, in_channels * 2, is_3d=True, kernel_size=1, padding=0, stride=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, kernel_size=3, padding=1, stride=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, kernel_size=3, padding=1, stride=1))

        self.feature_att_8 = FeatureAtt(in_channels * 2, 64)
        self.feature_att_16 = FeatureAtt(in_channels * 4, 192)
        self.feature_att_32 = FeatureAtt(in_channels * 6, 160)
        self.feature_att_up_16 = FeatureAtt(in_channels * 4, 192)
        self.feature_att_up_8 = FeatureAtt(in_channels * 2, 64)

    def forward(self, x, features):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)

        return conv


class IGEVStereo(nn.Module):
    def __init__(self, rbhm_pretrained=''):
        super().__init__()
        args = Map(MAX_DISP=192,
                   HIDDEN_DIMS=[128, 128, 128],
                   N_GRU_LAYERS=3,
                   N_DOWNSAMPLE=2,
                   SLOW_FAST_GRU=True,
                   CORR_LEVELS=2,
                   CORR_RADIUS=4,
                   TRAIN_ITERS=22,
                   VALID_ITERS=32,
                   )
        self.args = args
        self.max_disp = args.MAX_DISP

        context_dims = args.HIDDEN_DIMS

        self.cnet = MultiBasicEncoder(output_dim=[args.HIDDEN_DIMS, context_dims], norm_fn="batch",
                                      downsample=args.N_DOWNSAMPLE)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.HIDDEN_DIMS)

        self.context_zqr_convs = nn.ModuleList(
            [nn.Conv2d(context_dims[i], args.HIDDEN_DIMS[i] * 3, 3, padding=3 // 2) for i in
             range(self.args.N_GRU_LAYERS)])

        self.feature = Feature()

        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
        )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
        )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1), )
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24), nn.ReLU()
        )

        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1), )

        self.conv = BasicConv_IN(96, 96, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)

        self.corr_stem = BasicConv(8, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att = FeatureAtt(8, 96)
        self.cost_agg = hourglass(8)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

        self.height_head = Refinement(in_channels=4)
        pretrained_state_dict = torch.load(rbhm_pretrained, map_location='cpu')
        state_dict = {}
        for key, val in pretrained_state_dict.items():
            state_dict[key.replace('height_head.', '')] = val
        self.height_head.load_state_dict(state_dict)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def upsample_disp(self, disp, mask_feat_4, stem_2x):
        xspx = self.spx_2_gru(mask_feat_4, stem_2x)
        spx_pred = self.spx_gru(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        up_disp = context_upsample(disp * 4., spx_pred).unsqueeze(1)
        return up_disp

    def forward(self, data):
        image1 = data['left']
        image2 = data['right']
        """ Estimate disparity between pair of frames """
        test_mode = not self.training
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

        features_left = self.feature(image1)
        features_right = self.feature(image2)
        stem_2x = self.stem_2(image1)
        stem_4x = self.stem_4(stem_2x)
        stem_2y = self.stem_2(image2)
        stem_4y = self.stem_4(stem_2y)
        features_left[0] = torch.cat((features_left[0], stem_4x), 1)
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)

        match_left = self.desc(self.conv(features_left[0]))
        match_right = self.desc(self.conv(features_right[0]))
        gwc_volume = build_gwc_volume(match_left, match_right, self.args.MAX_DISP // 4, 8)
        gwc_volume = self.corr_stem(gwc_volume)
        gwc_volume = self.corr_feature_att(gwc_volume, features_left[0])
        geo_encoding_volume = self.cost_agg(gwc_volume, features_left)

        # Init disp from geometry encoding volume
        prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)
        init_disp = disparity_regression(prob, self.args.MAX_DISP // 4)

        del prob, gwc_volume

        if not test_mode:
            xspx = self.spx_4(features_left[0])
            xspx = self.spx_2(xspx, stem_2x)
            spx_pred = self.spx(xspx)
            spx_pred = F.softmax(spx_pred, 1)

        cnet_list = self.cnet(image1, num_layers=self.args.N_GRU_LAYERS)
        net_list = [torch.tanh(x[0]) for x in cnet_list]
        inp_list = [torch.relu(x[1]) for x in cnet_list]
        inp_list = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1)) for i, conv in
                    zip(inp_list, self.context_zqr_convs)]

        geo_block = Combined_Geo_Encoding_Volume
        geo_fn = geo_block(match_left.float(), match_right.float(), geo_encoding_volume.float(),
                           radius=self.args.CORR_RADIUS, num_levels=self.args.CORR_LEVELS)
        b, c, h, w = match_left.shape
        coords = torch.arange(w).float().to(match_left.device).reshape(1, 1, w, 1).repeat(b, h, 1, 1)
        disp = init_disp
        disp_preds = []

        # GRUs iterations to update disparity
        iters = self.args.VALID_ITERS if test_mode else self.args.TRAIN_ITERS
        for itr in range(iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords)
            if self.args.N_GRU_LAYERS == 3 and self.args.SLOW_FAST_GRU:  # Update low-res ConvGRU
                net_list = self.update_block(net_list, inp_list, iter16=True, iter08=False, iter04=False, update=False)
            if self.args.N_GRU_LAYERS >= 2 and self.args.SLOW_FAST_GRU:  # Update low-res ConvGRU and mid-res ConvGRU
                net_list = self.update_block(net_list, inp_list, iter16=self.args.N_GRU_LAYERS == 3, iter08=True,
                                             iter04=False, update=False)
            net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp,
                                                                  iter16=self.args.N_GRU_LAYERS == 3,
                                                                  iter08=self.args.N_GRU_LAYERS >= 2)

            disp = disp + delta_disp
            if test_mode and itr < iters - 1:
                continue

            # upsample predictions
            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)
            disp_preds.append(disp_up)

        pred_height = self.height_head(torch.cat([data["left"], disp_up], dim=1))
        pred_height = torch.sigmoid(pred_height) * 10
        if test_mode:
            return {'disp_pred': disp_up,
                    'pred_height': pred_height.squeeze(1)}

        init_disp = context_upsample(init_disp * 4., spx_pred.float()).unsqueeze(1)

        return {'init_disp': init_disp,
                'disp_preds': disp_preds,
                'disp_pred': disp_preds[-1],
                'pred_height': pred_height.squeeze(1)}

    def get_loss(self, model_pred, input_data):
        disp_gt = input_data["disp"]
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)
        dilated_bump_mask = input_data['dilated_bump_mask'].unsqueeze(1)
        valid = mask.float()

        disp_gt = disp_gt.unsqueeze(1)
        mag = torch.sum(disp_gt ** 2, dim=1).sqrt()
        valid = ((valid >= 0.5) & (mag < self.max_disp)).unsqueeze(1)
        assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
        assert not torch.isinf(disp_gt[valid.bool()]).any()

        disp_init_pred = model_pred['init_disp']
        disp_loss = 1.0 * F.smooth_l1_loss(disp_init_pred[valid.bool() * dilated_bump_mask], disp_gt[valid.bool() & dilated_bump_mask], reduction='mean')

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
            disp_loss += i_weight * i_loss[valid.bool() & dilated_bump_mask].mean()

        loss_info = {'scalar/train/loss_disp': disp_loss.item()}
        return disp_loss, loss_info
