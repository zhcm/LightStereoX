import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_

from stereo.utils import common_utils
from stereo.modeling.models.nmrf.utils.frame_utils import InputPadder, downsample_disp
from stereo.modeling.models.nmrf.config import configurable
from stereo.modeling.models.nmrf.matcher import bf_match
from stereo.modeling.models.nmrf.backbone import create_backbone
from stereo.modeling.models.nmrf.DPN import DPN
from stereo.modeling.models.nmrf.submodule import build_correlation_volume
from stereo.modeling.models.nmrf.NMP import (
    MLP,
    InferenceLayer,
    Inference,
    RefinementLayer,
    Refinement,
)
from stereo.modeling.models.rbhm.rbhm import Refinement as RbhmRefinement


class NMRF(nn.Module):
    @configurable
    def __init__(self,
                 backbone,
                 dpn,
                 num_proposals,
                 max_disp,
                 num_infer_layers,
                 num_refine_layers,
                 infer_embed_dim,
                 infer_n_heads,
                 mlp_ratio,
                 window_size,
                 refine_window_size,
                 with_refinement=True,
                 attn_drop=0.,
                 proj_drop=0.,
                 drop_path=0.,
                 dropout=0.,
                 return_intermediate=False,
                 normalize_before=False,
                 activation="gelu",
                 aux_loss=False,
                 divis_by=8,
                 compat=True,
                 criterion=None,
                 rbhm_pretrained=None):
        """
        aux_loss: True if auxiliary intermediate losses (losses at each encoder/decoder layer)
        """
        super().__init__()
        self.height_head = RbhmRefinement(in_channels=4)
        pretrained_state_dict = torch.load(rbhm_pretrained, map_location='cpu')
        state_dict = {}
        for key, val in pretrained_state_dict.items():
            state_dict[key.replace('height_head.', '')] = val
        self.height_head.load_state_dict(state_dict)

        self.num_proposals = num_proposals
        self.max_disp = max_disp
        self.aux_loss = aux_loss
        self.divis_by = divis_by
        self.criterion = criterion

        feat_dim = backbone.output_dim
        self.concatconv = nn.Sequential(
            nn.Conv2d(feat_dim, 128, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 1, 1, 0, bias=False))
        self.gw = nn.Sequential(
            nn.Conv2d(feat_dim, 128, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 1, 1, 0, bias=False))

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path, num_infer_layers)]
        infer_layers = nn.ModuleList([
            InferenceLayer(
                infer_embed_dim, mlp_ratio=mlp_ratio, window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2, n_heads=infer_n_heads,
                activation=activation,
                attn_drop=attn_drop, proj_drop=proj_drop, drop_path=dpr[i], dropout=dropout,
                normalize_before=normalize_before
            )
            for i in range(num_infer_layers)]
        )
        infer_norm = nn.LayerNorm(infer_embed_dim)
        self.inference = Inference(32, infer_embed_dim, layers=infer_layers, norm=infer_norm,
                                   return_intermediate=return_intermediate)
        self.infer_head = MLP(infer_embed_dim, infer_embed_dim, 8 * 8, 3)
        self.infer_score_head = nn.Linear(infer_embed_dim, 8 * 8)

        # init weights
        self.apply(self._init_weights)

        self.with_refinement = with_refinement
        if self.with_refinement:
            # refinement
            dpr = [x.item() for x in torch.linspace(0, drop_path, num_refine_layers)]
            refine_layers = nn.ModuleList([
                RefinementLayer(
                    infer_embed_dim, mlp_ratio=mlp_ratio, window_size=refine_window_size,
                    shift_size=0 if i % 2 == 0 else refine_window_size // 2, n_heads=infer_n_heads,
                    activation=activation,
                    attn_drop=attn_drop, proj_drop=proj_drop, drop_path=dpr[i], dropout=dropout,
                    normalize_before=normalize_before,
                )
                for i in range(num_refine_layers)]
            )
            refine_norm = nn.LayerNorm(infer_embed_dim)
            self.refinement = Refinement(32, infer_embed_dim, layers=refine_layers, norm=refine_norm,
                                       return_intermediate=return_intermediate)
            self.refine_head = MLP(infer_embed_dim, infer_embed_dim, 4 * 4, 3)

        self.dpn = dpn
        # backward compatible with the models released at CVPR'24
        self.compat = compat
        if compat:
            self.backbone = backbone
        else:
            self.image_encoder = backbone

        # to keep track of which device the nn.Module is on
        self.register_buffer("device_indicator_tensor", torch.empty(0))

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    @classmethod
    def from_config(cls, cfg):
        # backbone
        backbone = create_backbone(cfg)

        # disparity proposal network
        dpn = DPN(cfg)

        return {
            "backbone": backbone,
            "dpn": dpn,
            "num_proposals": cfg.DPN.NUM_PROPOSALS,
            "max_disp": cfg.DPN.MAX_DISP,
            "aux_loss": cfg.SOLVER.AUX_LOSS,
            "num_infer_layers": cfg.NMP.NUM_INFER_LAYERS,
            "num_refine_layers": cfg.NMP.NUM_REFINE_LAYERS,
            "infer_embed_dim": cfg.NMP.INFER_EMBED_DIM,
            "infer_n_heads": cfg.NMP.INFER_N_HEADS,
            "mlp_ratio": cfg.NMP.MLP_RATIO,
            "window_size": cfg.NMP.WINDOW_SIZE,
            "refine_window_size": cfg.NMP.REFINE_WINDOW_SIZE,
            "attn_drop": cfg.NMP.ATTN_DROP,
            "proj_drop": cfg.NMP.PROJ_DROP,
            "drop_path": cfg.NMP.DROP_PATH,
            "dropout": cfg.NMP.DROPOUT,
            "normalize_before": cfg.NMP.NORMALIZE_BEFORE,
            "return_intermediate": cfg.NMP.RETURN_INTERMEDIATE,
            "divis_by": cfg.DATASETS.DIVIS_BY,
            "compat": cfg.BACKBONE.COMPAT,
        }

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.InstanceNorm2d)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.device_indicator_tensor.device

    def extract_feature(self, img1, img2):
        img_batch = torch.cat((img1, img2), dim=0)  # [2B, C, H, W]
        if self.compat:
            features = self.backbone(img_batch)
        else:
            features = self.image_encoder(img_batch)  # list of [2B, C, H, W], resolution from high to low

        # reverse resolution from low to high
        features = features[::-1]

        # split to list of tuple, res from low to high
        features = [torch.chunk(feature, 2, dim=0) for feature in features]

        feature1, feature2 = map(list, zip(*features))

        return feature1, feature2

    def forward(self, sample):
        """
        It returns a dict with the following elements:
            - "proposal": disparity proposals, tensor of dim [n, B*H/2*W/2, num_proposals]
            - "prob": disparity candidate probability, tensor of dim [B*H/2*W/2, W/2]
            - "initial_proposal": disparity proposals from initialization stage, tensor of dim [B*H/2*W/2, num_proposals]
            - "disp": disparity prediction, tensor of dim [B, H, W]
            - "aux_outputs": Optional, only returned when auxiliary losses are activated. It is a list of
                             dictionaries containing the four above keys for each intermediate layer.
        """
        image1 = sample['left'].to(self.device)
        image2 = sample['right'].to(self.device)

        # We assume the input padding is not needed during training by setting adequate crop size
        if not self.training:
            padder = InputPadder(image1.shape, mode='proposal', divis_by=self.divis_by)
            image1, image2 = padder.pad(image1, image2)
        fmap1_list, fmap2_list = self.extract_feature(image1, image2)
        cost_volume = build_correlation_volume(fmap1_list[0], fmap2_list[0], self.max_disp // 8, self.dpn.cost_group)  # [B,G,D,H,W]
        cost_volume, prob, label_seeds, labels = self.dpn(cost_volume, fmap1_list)

        # ---- NMRF inference ---- #
        fmap1 = self.concatconv(fmap1_list[0])
        fmap2 = self.concatconv(fmap2_list[0])
        fmap1_gw = self.gw(fmap1_list[0])
        fmap2_gw = self.gw(fmap2_list[0])
        labels_curr = labels[-1].detach()

        tgt = self.inference(labels_curr, fmap1, fmap2, fmap1_gw, fmap2_gw)
        disp_delta = self.infer_head(tgt)  # [num_aux_layers,BHW,N,8*8]
        coarse_disp = F.relu(labels_curr[None].unsqueeze(-1) + disp_delta)
        mask = .25 * self.infer_score_head(tgt)  # [num_aux_layers,BHW,N,8*8]
        bs, _, ht, wd = fmap1.shape
        coarse_disp = rearrange(coarse_disp, 'a (b h w) n (hs ws) -> a b (h hs) (w ws) n', h=ht, w=wd, hs=8).contiguous()
        mask = rearrange(mask, 'a (b h w) n (hs ws) -> a b (h hs) (w ws) n', h=ht, w=wd, hs=8)

        disp_pred = None
        if self.with_refinement:
            # refinement
            _, indices = torch.max(mask[-1], dim=-1, keepdim=True)
            disp_curr = torch.gather(coarse_disp[-1], dim=-1, index=indices).squeeze(-1) * 2  # [B,H,W]
            disp_curr = rearrange(disp_curr, 'b (h hs) (w ws) -> b h w (hs ws)', hs=4, ws=4)
            disp_curr = torch.median(disp_curr, dim=-1, keepdim=False)[0]
            disp_curr = disp_curr.detach()
            fmap1 = self.concatconv(fmap1_list[1])
            fmap2 = self.concatconv(fmap2_list[1])
            fmap1_gw = self.gw(fmap1_list[1])
            fmap2_gw = self.gw(fmap2_list[1])
            tgt = self.refinement(disp_curr, fmap1, fmap2, fmap1_gw, fmap2_gw)
            disp_delta = self.refine_head(tgt)  # [num_aux_layers,BHW,4*4]
            bs, _, ht, wd = fmap1.shape
            disp_delta = rearrange(disp_delta, 'a (b h w) p -> a b h w p', h=ht, w=wd)
            disp_pred = F.relu(disp_curr[None].unsqueeze(-1) + disp_delta)
            disp_pred = rearrange(disp_pred, 'a b h w (hs ws) -> a b (h hs) (w ws)', hs=4).contiguous()

        if disp_pred is not None:
            disp = disp_pred[-1] * 4
        else:
            _, indices = torch.max(mask[-1], dim=-1, keepdim=True)
            disp = torch.gather(coarse_disp[-1], dim=-1, index=indices).squeeze(-1) * 8  # [B,H,W]

        if not self.training:
            disp = padder.unpad(disp.unsqueeze(1)).squeeze(1)

        bs = image1.shape[0]
        label_seeds = label_seeds.reshape(bs, -1, self.num_proposals)
        proposal = labels[-1].reshape(bs, -1, self.num_proposals)
        out = {'proposal': proposal, 'prob': prob, 'initial_proposal': label_seeds, 'disp': disp}
        if disp_pred is not None:
            out['disp_pred'] = disp_pred[-1]
        if self.aux_loss and self.training:
            out['aux_outputs'] = self._set_aux_loss(disp_pred, coarse_disp, mask)

        if not self.training:
            out['disp_pred'] = out['disp']

        pred_height = self.height_head(torch.cat([sample['left'], out['disp_pred'].unsqueeze(1)], dim=1))
        pred_height = torch.sigmoid(pred_height) * 10
        out['pred_height'] = pred_height.squeeze(1)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, disp_pred, coarse_disp, logits_pred):
        res = []
        for coarse_disp_i, logits_pred_i in zip(coarse_disp, logits_pred):
            res.append(dict(disp_pred=coarse_disp_i, logits_pred=logits_pred_i))
        if disp_pred is None:
            return res
        for disp_pred_i in disp_pred[:-1]:
            res.append(dict(disp_pred=disp_pred_i))
        return res

    def get_loss(self, model_pred, input_data):
        loss_dict = self.criterion(model_pred, input_data, log=True)
        weight_dict = self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_info = {'scalar/train/loss_disp': losses.item()}

        imagenet_rgb_mean = torch.tensor([123.675, 116.28, 103.53]).cuda().unsqueeze(dim=1).unsqueeze(dim=2)
        imagenet_rgb_std = torch.tensor([58.395, 57.12, 57.375]).cuda().unsqueeze(dim=1).unsqueeze(dim=2)
        loss_info['image/train/image'] = torch.cat([input_data['left'][0] * imagenet_rgb_std + imagenet_rgb_mean, input_data['right'][0] * imagenet_rgb_std + imagenet_rgb_mean], dim=1) / 256
        loss_info['image/train/disp'] = common_utils.color_map_tensorboard(input_data['disp'][0], model_pred['disp_pred'].squeeze(1)[0])

        return losses, loss_info


class Criterion(nn.Module):
    """ This class computes the loss for disparity proposal extraction.
    The process happens in two steps:
        1) we compute a one-to-one matching between ground truth disparities and the outputs of the model
        2) we supervise each output to be closer to the ground truth disparity it was matched to

    Note: to avoid trivial solution, we add a prior term in the loss computation that we favor positive output.
    """
    def __init__(self, weight_dict, max_disp, loss_type):
        """ Create the criterion.
        Parameters:
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
        """
        super().__init__()

        self.weight_dict = weight_dict
        self.max_disp = max_disp
        if loss_type == "SMOOTH_L1":
            self.loss_fn = F.smooth_l1_loss
        else:
            self.loss_fn = F.l1_loss

    def loss_prop(self, disp_prop, gt_disp):
        tgt_disp = gt_disp.clone()
        # ground truth modes larger than 320 are ignored in following sort and matching
        tgt_disp[tgt_disp >= 320] = 0
        dist = (tgt_disp[:, :, None] - disp_prop[:, None, :]).abs()
        dist[tgt_disp == 0, :] = 1e6  # avoid assigned to null gt
        dist = torch.min(dist, dim=-1, keepdim=False)[0]
        indices = dist.argsort(dim=-1)
        # sort ground truth modes based on distance to the proposal set
        tgt_disp = torch.gather(tgt_disp, dim=-1, index=indices)

        # nms
        for i in range(3):
            dist = (tgt_disp[:, i+1:] - tgt_disp[:, i:i+1]).abs()
            mask = (tgt_disp[:, i:i+1] > 0) & (dist < 8)
            tgt_disp[:, i+1:][mask] = 0

        # Retrieve the bipartite matching between proposals and ground truth modes
        num_seed = disp_prop.shape[1]
        if num_seed < 4:
            disp_prop_pad = tgt_disp.clone()
            disp_prop_pad[:, :num_seed] = disp_prop
            disp_prop = disp_prop_pad
        indices, disp_error = bf_match(disp_prop, tgt_disp)
        src_disp = torch.gather(disp_prop, dim=1, index=indices)

        mask = (tgt_disp > 0) & (tgt_disp < self.max_disp)
        total_gts = torch.sum(mask)
        # disparity loss for matched predictions
        loss_disp = F.smooth_l1_loss(src_disp[mask], tgt_disp[mask], reduction='sum')
        losses = {'proposal_disp': loss_disp / (total_gts + 1e-6)}

        # for logging
        valid_pixs = disp_error[disp_error != 1e5]
        losses['disp_error'] = valid_pixs.sum() / (valid_pixs.numel() + 1e-6)

        return losses

    @staticmethod
    def loss_init(prob, gt_disp, occlusion_map, occlusion_map_2):
        nd = prob.shape[-1]
        bs, ht, wd = gt_disp.shape
        gt_disp = torch.clamp(gt_disp, min=0)
        valid = (gt_disp > 0) & (gt_disp < 320) & (~occlusion_map)

        ref = torch.arange(0, wd, dtype=torch.int64, device=prob.device).reshape(1, 1, -1).repeat(bs, ht, 1)
        coord = ref - gt_disp  # corresponding coordinate in the right view
        valid = torch.logical_and(valid, coord >= 0)  # correspondence should within image boundary
        coord = torch.clamp(torch.floor(coord), min=0).to(torch.int64)
        nocc_map_2 = ~occlusion_map_2
        nocc_map_2 = torch.gather(nocc_map_2, dim=-1, index=coord)  # whether correspondence is occluded

        valid = torch.logical_and(valid, nocc_map_2)

        # scale ground-truth disparities
        tgt_disp = gt_disp / 8

        weights = torch.ones_like(tgt_disp)
        weights[~valid] = 0

        tgt_disp = rearrange(tgt_disp, 'b (h m) (w n) -> (b h w) (m n)', m=8, n=8)
        weights = rearrange(weights, 'b (h m) (w n) -> (b h w) (m n)', m=8, n=8)
        valid = rearrange(valid, 'b (h m) (w n) -> (b h w) (m n)', m=8, n=8)

        lower_bound = torch.floor(tgt_disp).to(torch.int64)
        high_bound = lower_bound + 1
        high_prob = tgt_disp - lower_bound
        lower_bound = torch.clamp(lower_bound, max=nd - 1)
        high_bound = torch.clamp(high_bound, max=nd - 1)

        lower_prob = (1 - high_prob) * weights
        high_prob = high_prob * weights

        label = torch.zeros_like(prob)
        label.scatter_reduce_(dim=-1, index=lower_bound, src=lower_prob, reduce="sum")
        label.scatter_reduce_(dim=-1, index=high_bound, src=high_prob, reduce="sum")

        # normalize weights
        normalizer = torch.clamp(torch.sum(label, dim=-1, keepdim=True), min=1e-3)
        label = label / normalizer

        mask = label > 0
        log_prob = -(torch.log(torch.clamp(prob[mask], min=1e-6)) * label[mask]).sum()
        valid_pixs = (valid.float().sum(dim=-1) > 0).sum()

        losses = {'init': log_prob / (valid_pixs + 1e-6)}
        assert not torch.any(torch.isnan(losses['init']))
        return losses

    def loss_coarse(self, disp_pred, logits_pred, disp_gt):
        mask = (disp_gt > 0) & (disp_gt < self.max_disp)
        prob = F.softmax(logits_pred, dim=-1)
        disp_gt = disp_gt.unsqueeze(-1).expand_as(disp_pred)
        error = self.loss_fn(disp_pred, disp_gt, reduction='none')
        if torch.any(mask):
            loss = torch.sum(prob * error, dim=-1, keepdim=False)[mask].mean()
        else:  # dummy loss
            loss = F.smooth_l1_loss(disp_pred, disp_pred.detach(), reduction='mean') + F.smooth_l1_loss(logits_pred, logits_pred.detach(), reduction='mean')
        return {"loss_coarse_disp": loss}

    def loss_disp(self, disp_pred, disp_gt):
        mask = (disp_gt > 0) & (disp_gt < self.max_disp)
        if torch.any(mask):
            loss = self.loss_fn(disp_pred[mask], disp_gt[mask], reduction='mean')
        else:
            loss = F.smooth_l1_loss(disp_pred, disp_pred.detach(), reduction='mean')
        return {"loss_disp": loss}

    def forward(self, outputs, targets, log=True):
        """This performs the loss computation.
        outputs: dict of tensors, see the output specification of the model for the format
        targets: dict of tensors, the expected keys in each dict depends on the losses applied.
            - "disp": [batch_size, H_I, W_I],
            - "occlusion_map": boolean tensor [batch_size, H_I, W] occlusion map of left image,
            - "occlusion_map_2": boolean tensor [batch_size, H_I, W] occlusion map of right image.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        prob = outputs_without_aux['prob']  # [B*H*W,D]
        disp_prop = outputs_without_aux['proposal'] * 8  # [B,H*W,N]
        disp_prop = rearrange(disp_prop, 'b l n -> (b l) n')  # [B*H*W,N]
        disp = outputs_without_aux['disp']
        device = disp.device

        tgt_disp = targets['disp'].to(device)
        valid = targets['valid'].to(device)
        tgt_disp[~valid] = 0
        occlusion_map = targets['occ_mask'].to(device)
        occlusion_map_2 = targets['occ_mask_2'].to(device)
        label = targets['super_pixel_label'].to(device)
        tgt_disp_mini = downsample_disp(tgt_disp, label)
        bs, ht = tgt_disp_mini.shape[:2]
        tgt_disp_mini = tgt_disp_mini.reshape(-1, 4)  # [B*H*W, 4]

        losses = self.loss_prop(disp_prop, tgt_disp_mini)
        losses.update(self.loss_init(prob, tgt_disp, occlusion_map, occlusion_map_2))
        if 'disp_pred' in outputs_without_aux:
            disp_pred = outputs_without_aux['disp_pred'] * 4
            losses.update(self.loss_disp(disp_pred, tgt_disp))
        if log:
            valid = (tgt_disp > 0) & (tgt_disp < self.max_disp)
            err = torch.abs(disp - tgt_disp)
            losses['epe_train'] = err[valid].mean()

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if 'logits_pred' in aux_outputs:
                    disp_pred = aux_outputs['disp_pred'] * 8
                    logits_pred = aux_outputs['logits_pred']
                    l_dict = self.loss_coarse(disp_pred, logits_pred, tgt_disp)
                else:
                    disp_pred = aux_outputs['disp_pred'] * 4
                    l_dict = self.loss_disp(disp_pred, tgt_disp)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


def build(cfg):
    model = NMRF(cfg)
    weight_dict = {'proposal_disp': 1, 'init': 1}
    assert len(cfg.SOLVER.LOSS_WEIGHTS) == cfg.NMP.NUM_INFER_LAYERS + cfg.NMP.NUM_REFINE_LAYERS
    if cfg.SOLVER.AUX_LOSS:
        aux_weight_dict = {}
        for i in range(cfg.NMP.NUM_INFER_LAYERS + cfg.NMP.NUM_REFINE_LAYERS-1):
            if i < cfg.NMP.NUM_INFER_LAYERS:
                aux_weight_dict.update({f'loss_coarse_disp_{i}': cfg.SOLVER.LOSS_WEIGHTS[i]})
            else:
                aux_weight_dict.update({f'loss_disp_{i}': cfg.SOLVER.LOSS_WEIGHTS[i]})
        weight_dict.update(aux_weight_dict)
    weight_dict.update({'loss_disp': cfg.SOLVER.LOSS_WEIGHTS[-1]})
    criterion = Criterion(weight_dict, cfg)

    return model, criterion
