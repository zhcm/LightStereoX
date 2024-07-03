import torch
import torch.nn.functional as F
from torch import nn, Tensor


class RegressionHead(nn.Module):
    """
    Regress disparity and occlusion mask
    """

    def __init__(self, ot: bool = True):
        super(RegressionHead, self).__init__()
        self.ot = ot
        self.phi = nn.Parameter(torch.tensor(0.0, requires_grad=True))  # dustbin cost

    @staticmethod
    def sinkhorn(attn: Tensor, log_mu: Tensor, log_nu: Tensor, iters: int):
        """
        Sinkhorn Normalization in Log-space as matrix scaling problem.
        Regularization strength is set to 1 to avoid manual checking for numerical issues
        Adapted from SuperGlue (https://github.com/magicleap/SuperGluePretrainedNetwork)
        :param attn: input attention weight, [bz,H/downsampling,W/downsampling+1,W/downsampling+1]
        :param log_mu: marginal distribution of left image, [bz,H/downsampling,W/downsampling+1]
        :param log_nu: marginal distribution of right image, [bz,H/downsampling,W/downsampling+1]
        :param iters: number of iterations
        :return: updated attention weight
        """
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for idx in range(iters):
            # scale v first then u to ensure row sum is 1, col sum slightly larger than 1
            v = log_nu - torch.logsumexp(attn + u.unsqueeze(3), dim=2)
            u = log_mu - torch.logsumexp(attn + v.unsqueeze(2), dim=3)

        return attn + u.unsqueeze(3) + v.unsqueeze(2)

    def optimal_transport(self, attn: Tensor, iters: int):
        """
        Perform Differentiable Optimal Transport in Log-space for stability
        Adapted from SuperGlue (https://github.com/magicleap/SuperGluePretrainedNetwork)
        :param attn: raw attention weight, [bz, H/downsampling, W/downsampling, W/downsampling]
        :param iters: number of iterations to run sinkhorn
        :return: updated attention weight, [bz, H/downsampling, W/downsampling+1, W/downsampling+1]
        """
        bs, h, w, _ = attn.shape
        # set marginal to be uniform distribution
        marginal = torch.cat([torch.ones([w]), torch.tensor([w]).float()]) / (2 * w)
        log_mu = marginal.log().to(attn.device).expand(bs, h, w + 1)
        log_nu = marginal.log().to(attn.device).expand(bs, h, w + 1)
        # add dustbins
        similarity_matrix = torch.cat([attn, self.phi.expand(bs, h, w, 1).to(attn.device)], -1)
        similarity_matrix = torch.cat([similarity_matrix, self.phi.expand(bs, h, 1, w + 1).to(attn.device)], -2)
        # sinkhorn
        attn_ot = self.sinkhorn(similarity_matrix, log_mu, log_nu, iters)
        # convert back from log space, recover probabilities by normalization 2W
        attn_ot = (attn_ot + torch.log(torch.tensor([2.0 * w]).to(attn.device))).exp()
        return attn_ot

    def softmax(self, attn: Tensor):
        """
        Alternative to optimal transport
        :param attn: raw attention weight, [bz,H/downsampling,W/downsampling,W/downsampling]
        :return: updated attention weight, [bz,H/downsampling,W/downsampling+1,W/downsampling+1]
        """
        bs, h, w, _ = attn.shape
        # add dustbins
        similarity_matrix = torch.cat([attn, self.phi.expand(bs, h, w, 1).to(attn.device)], -1)
        similarity_matrix = torch.cat([similarity_matrix, self.phi.expand(bs, h, 1, w + 1).to(attn.device)], -2)

        attn_softmax = F.softmax(similarity_matrix, dim=-1)
        return attn_softmax

    @staticmethod
    def compute_unscaled_pos_shift(w: int, device: torch.device):
        """
        Compute relative difference between each pixel location from left image to right image, to be used to calculate disparity
        :param w: image width//downsampling
        :param device: torch device
        :return: relative pos shifts
        """
        pos_r = torch.linspace(0, w - 1, w)[None, None, None, :].to(device)  # [1, 1, 1, w]
        pos_l = torch.linspace(0, w - 1, w)[None, None, :, None].to(device)  # [1, 1, w, 1]
        pos = pos_l - pos_r  # [1, 1, w, w]
        pos[pos < 0] = 0
        return pos

    @staticmethod
    def compute_low_res_disp(pos_shift: Tensor, attn_weight: Tensor, occ_mask: Tensor):
        """
        Compute low res disparity using the attention weight by finding the most attended pixel and regress within the 3px window
        :param pos_shift: relative pos shift (computed from compute_unscaled_pos_shift), [1, 1, W/downsampling, W/downsampling]
        :param attn_weight: attention (computed from optimal_transport), [bz, H/downsampling, W/downsampling, W/downsampling]
        :param occ_mask: ground truth occlusion mask, [bz, H/downsampling, W/downsampling]
        :return: low res disparity, [N,H/downsampling,W/downsampling] and attended similarity sum, [N,H/downsampling,W/downsampling]
        """

        # find high response area
        high_response = torch.argmax(attn_weight, dim=-1)  # [bz, H/downsampling, W/downsampling]

        # build 3 px local window, [bz, H/downsampling, W/downsampling, 3]
        response_range = torch.stack([high_response - 1, high_response, high_response + 1], dim=-1)

        # attention with re-weighting, # [bz, H/downsampling, WL/downsampling, WR/downsampling + 2]
        attn_weight_pad = F.pad(attn_weight, [1, 1], value=0.0)
        # [bz, H/downsampling, WL/downsampling, 3]
        attn_weight_rw = torch.gather(attn_weight_pad, -1, response_range + 1)

        # compute sum of attention
        norm = attn_weight_rw.sum(-1, keepdim=True)
        if occ_mask is None:
            norm[norm < 0.1] = 1.0
        else:
            norm[occ_mask, :] = 1.0  # set occluded region norm to be 1.0 to avoid division by 0

        # re-normalize to 1
        attn_weight_rw = attn_weight_rw / norm  # re-sum to 1
        pos_pad = F.pad(pos_shift, [1, 1]).expand_as(attn_weight_pad)
        pos_rw = torch.gather(pos_pad, -1, response_range + 1)

        # compute low res disparity
        disp_pred_low_res = (attn_weight_rw * pos_rw)  # [bz, h, w, 3]

        return disp_pred_low_res.sum(-1), norm

    def forward(self, attn_weight, occ_mask):
        """
        :param attn_weight: raw attention weight, [bz, H/downsampling, W/downsampling, W/downsampling]
        :param occ_mask: [bz, H/downsampling, W/downsampling]
        :return: dictionary of predicted values
        """
        bs, h, w, _ = attn_weight.size()

        # normalize attention to 0-1
        if self.ot:
            attn_ot = self.optimal_transport(attn_weight, 10)
        else:
            attn_ot = self.softmax(attn_weight)

        pos_shift = self.compute_unscaled_pos_shift(w, attn_ot.device)
        disp_pred_low_res, matched_attn = self.compute_low_res_disp(pos_shift, attn_ot[..., :-1, :-1], occ_mask)
        occ_pred_low_res = (1.0 - matched_attn).squeeze(-1)

        # [bz, h, w], [bz, h, w], [bz, h, w+1, w+1]
        return disp_pred_low_res, occ_pred_low_res, attn_ot
