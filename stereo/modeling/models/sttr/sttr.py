# @Time    : 2023/11/27 15:36
# @Author  : zhangchenming
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sttr_utils.tokenizer import Tokenizer
from .sttr_utils.pos_encoder import PositionEncodingSine1DRelative
from .sttr_utils.transformer import Transformer
from .sttr_utils.regression_head import RegressionHead
from .sttr_utils.context_adjustment_layer import ContextAdjustmentLayer
from .sttr_utils.utils import torch_1d_sample


class STTR(nn.Module):
    def __init__(self, backbone, downsample=3, validation_max_disp=-1, px_threshold=3, channel_dim=128, position_encoding='sine1d_rel', nhead=8, num_attn_layers=6,
                 regression_head='ot', cal_num_blocks=8, cal_feat_dim=16, cal_expansion_ratio=4):
        super(STTR, self).__init__()
        self.validation_max_disp = validation_max_disp
        self.px_threshold = px_threshold
        self.downsample = downsample

        self.sampled_cols = None
        self.sampled_rows = None

        self.backbone = backbone
        out_dims = self.backbone.out_dims
        backbone_feat_channel = [out_dims['scale4'], out_dims['scale3'], out_dims['scale2']]
        self.tokenizer = Tokenizer(block_config=[4, 4, 4, 4],
                                   backbone_feat_channel=backbone_feat_channel,
                                   hidden_dim=channel_dim,
                                   growth_rate=4)

        if position_encoding == 'sine1d_rel':
            self.pos_encoder = PositionEncodingSine1DRelative(channel_dim)
        else:
            self.pos_encoder = None

        self.transformer = Transformer(hidden_dim=channel_dim,
                                       nhead=nhead,
                                       num_attn_layers=num_attn_layers)
        self.regression_head = RegressionHead(regression_head == 'ot')

        self.cal = ContextAdjustmentLayer(cal_num_blocks, cal_feat_dim, cal_expansion_ratio)

        self.reset_parameters()

    def reset_parameters(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def forward(self, data):
        left = data['left']
        right = data['right']
        occ_mask = data['occ_mask'].bool()
        occ_mask_right = data['occ_mask_right'].bool()
        bz, _, h, w = left.size()

        src_stereo = torch.cat([left, right], dim=0)
        # [2bz, 128, H/16, W/16], [2bz, 128, H/8, W/8], [2bz, 64, H/4, W/4], [2bz, 3, H, W]
        features = self.backbone(src_stereo)
        t_inputs = [features['scale4'], features['scale3'], features['scale2'], src_stereo]
        tokens = self.tokenizer(t_inputs)  # [2bz, C, H, W]

        # 下采样
        if self.downsample > 0:
            offset = int(self.downsample / 2)
            self.sampled_cols = torch.arange(offset, w, self.downsample).to(left.device)  # [w]
            self.sampled_rows = torch.arange(offset, h, self.downsample).to(left.device)  # [h]
            tokens = tokens[:, :, :, self.sampled_cols]
            tokens = tokens[:, :, self.sampled_rows, :]
            occ_mask = occ_mask[:, :, self.sampled_cols]
            occ_mask = occ_mask[:, self.sampled_rows, :]
            occ_mask_right = occ_mask_right[:, :, self.sampled_cols]
            occ_mask_right = occ_mask_right[:, self.sampled_rows, :]
            w = self.sampled_cols.size(-1)
            h = self.sampled_rows.size(-1)
            scale = left.size(-1) / float(self.sampled_cols.size(-1))
        else:
            scale = 1.0

        feat_left = tokens[:bz]  # [bz, C, h, w]
        feat_right = tokens[bz:]  # [bz, C, h, w]

        # 位置编码
        if self.pos_encoder is not None:
            pos_enc = self.pos_encoder(w, scale, left.device)  # [2*w-1, C]
            with torch.no_grad():
                indexes_r = torch.linspace(w - 1, 0, w).view(w, 1).to(left.device)
                indexes_c = torch.linspace(0, w - 1, w).view(1, w).to(left.device)
                pos_indexes = (indexes_r + indexes_c).view(-1).long()  # [w, w] -> [w*w]
            pos_enc = torch.index_select(pos_enc, 0, pos_indexes).view(w, w, -1)  # [w*w, C] -> [w, w, C]
        else:
            pos_enc = None

        # attention
        attn_weight = self.transformer(feat_left, feat_right, pos_enc)

        # 视差回归 [bz, h, w], [bz, h, w], [bz, h, w+1, w+1]
        disp_pred_low_res, occ_pred_low_res, attn_ot = self.regression_head(attn_weight, occ_mask)

        if self.downsample > 0:
            disp_pred_low_res = disp_pred_low_res * scale
            # [bz, 1, H, W]
            disp_pred = F.interpolate(disp_pred_low_res[:, None], size=(left.size()[2], left.size()[3]), mode='nearest')
            # [bz, 1, H, W]
            occ_pred = F.interpolate(occ_pred_low_res[:, None], size=(left.size()[2], left.size()[3]), mode='nearest')
            # refine
            mean_disp_pred = disp_pred.mean()
            std_disp_pred = disp_pred.std() + 1e-6
            disp_pred_normalized = (disp_pred - mean_disp_pred) / std_disp_pred
            occ_pred_normalized = (occ_pred - 0.5) / 0.5
            disp_pred_normalized, occ_pred = self.cal(disp_pred_normalized, occ_pred_normalized, left)
            occ_pred = occ_pred.squeeze(1)
            disp_pred = disp_pred_normalized * std_disp_pred + mean_disp_pred
            disp_pred = disp_pred.squeeze(1)
        else:
            disp_pred = disp_pred_low_res
            occ_pred = occ_pred_low_res

        result = {'disp_pred': disp_pred,  # [bz, H, W]
                  'disp_pred_low_res': disp_pred_low_res,  # [bz, h, w]
                  'occ_pred': occ_pred  # [bz, H, W]
                  }

        if self.training:
            disp_gt = data['disp']  # [bz, H, W]
            _, _, w = disp_gt.size()
            pos_l = torch.linspace(0, w - 1, w)[None,].to(disp_gt.device)  # [1, W]
            target = (pos_l - disp_gt)[..., None]  # [bz, H, W, 1]
            if self.downsample > 0:
                target = target[:, :, self.sampled_cols, :]
                target = target[:, self.sampled_rows, :, :]
            target = target / scale  # [bz, h, w, 1]

            gt_response = torch_1d_sample(attn_ot[..., :-1, :-1], target, 'linear')  # [bz, h, w]
            gt_response_occ_left = attn_ot[..., :-1, -1][occ_mask]
            gt_response_occ_right = attn_ot[..., -1, :-1][occ_mask_right]

            result.update({'gt_response': gt_response,  # [bz, h, w]
                           'gt_response_occ_left': gt_response_occ_left,  # [num1, ]
                           'gt_response_occ_right': gt_response_occ_right,  # [num2, ]
                           })

        return result

    def get_loss(self, model_pred, input_data):

        disp_gt = input_data['disp']  # [bz, H, W]
        occ_mask_gt = input_data['occ_mask'].bool()  # [bz, H, W]

        disp_pred = model_pred['disp_pred']  # [bz, H, W]
        occ_pred = model_pred['occ_pred']  # [bz, H, W]
        disp_pred_low_res = model_pred['disp_pred_low_res']  # [bz, h, w]

        # [bz, H, W]
        if self.validation_max_disp == -1:
            invalid_mask = disp_gt <= 0.0
        else:
            invalid_mask = torch.logical_or(disp_gt <= 0.0, disp_gt >= self.validation_max_disp)

        rr_loss = self.compute_rr_loss(model_pred, invalid_mask)
        l1_lowres_loss = self.compute_l1_loss(disp_pred_low_res, disp_gt, invalid_mask, False)
        l1_loss = self.compute_l1_loss(disp_pred, disp_gt, invalid_mask, True)
        occ_be_loss = self.compute_entropy_loss(occ_pred, occ_mask_gt, invalid_mask)
        all_loss = rr_loss + l1_lowres_loss + l1_loss + occ_be_loss

        with torch.no_grad():
            error_px = torch.sum(torch.abs(disp_pred[~invalid_mask] - disp_gt[~invalid_mask]) > self.px_threshold).item()
            total_px = torch.sum(~invalid_mask).item()
            epe = nn.L1Loss()(disp_pred[~invalid_mask], disp_gt[~invalid_mask])

        pred_mask: torch.Tensor = occ_pred > 0.5
        inter_occ = torch.logical_and(pred_mask, occ_mask_gt).sum()
        union_occ = torch.logical_or(torch.logical_and(pred_mask, ~invalid_mask), occ_mask_gt).sum()
        inter_noc = torch.logical_and(~pred_mask, ~invalid_mask).sum()
        union_noc = torch.logical_or(torch.logical_and(~pred_mask, occ_mask_gt), ~invalid_mask).sum()
        iou = (inter_occ + inter_noc).float() / (union_occ + union_noc)

        info = {'scalar/train/loss/all_loss': all_loss.item(),
                'scalar/train/loss/rr_loss': rr_loss.item(),
                'scalar/train/loss/l1_lowres_loss': l1_lowres_loss.item(),
                'scalar/train/loss/l1_loss': l1_loss.item(),
                'scalar/train/loss/occ_be_loss': occ_be_loss.item(),
                'scalar/train/error_px': error_px,
                'scalar/train/total_px': total_px,
                'scalar/train/epe': epe,
                'scalar/train/iou': iou}

        return all_loss, info

    def compute_rr_loss(self, model_pred, invalid_mask):
        if self.downsample > 0:
            invalid_mask = invalid_mask[:, :, self.sampled_cols]
            invalid_mask = invalid_mask[:, self.sampled_rows]
        rr_loss = - torch.log(model_pred['gt_response'] + 1e-6)
        rr_loss = rr_loss[~invalid_mask]
        rr_loss_occ_left = - torch.log(model_pred['gt_response_occ_left'] + 1e-6)
        rr_loss_occ_right = - torch.log(model_pred['gt_response_occ_right'] + 1e-6)
        rr_loss = torch.cat([rr_loss, rr_loss_occ_left, rr_loss_occ_right])
        rr_loss = rr_loss.mean()
        return rr_loss

    def compute_l1_loss(self, disp_pred, disp_gt, invalid_mask, fullres):
        if not fullres:
            if self.downsample > 0:
                invalid_mask = invalid_mask[:, :, self.sampled_cols]
                invalid_mask = invalid_mask[:, self.sampled_rows]
                disp_gt = disp_gt[:, :, self.sampled_cols]
                disp_gt = disp_gt[:, self.sampled_rows]
        return nn.SmoothL1Loss()(disp_pred[~invalid_mask], disp_gt[~invalid_mask])

    @staticmethod
    def compute_entropy_loss(occ_pred, occ_mask_gt, invalid_mask):
        eps = 1e-6
        entropy_loss_occ = -torch.log(occ_pred[occ_mask_gt] + eps)
        entropy_loss_noc = - torch.log(1.0 - occ_pred[~invalid_mask] + eps)
        entropy_loss = torch.cat([entropy_loss_occ, entropy_loss_noc])
        return entropy_loss.mean()


def get_model_params(model, base_lr):
    if model.__class__.__name__ == 'DistributedDataParallel':
        vanilla_model = model.module
    else:
        vanilla_model = model

    higher_lr_params = list(vanilla_model.regression_head.parameters()) + list(vanilla_model.cal.parameters())
    higher_lr_set = set(higher_lr_params)
    default_lr_params = [p for p in vanilla_model.parameters() if p not in higher_lr_set]

    opt_params = [
        {'params': [p for p in default_lr_params if p.requires_grad], 'lr': base_lr},
        {'params': [p for p in higher_lr_params if p.requires_grad], 'lr': base_lr * 2}
    ]

    return opt_params
