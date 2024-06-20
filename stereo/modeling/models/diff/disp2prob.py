import torch


def disp2prob(max_disp, disp_gt):
    assert disp_gt.dim() == 4

    b, c, h, w = disp_gt.shape
    assert c == 1  # B x 1 x H x W

    index = torch.linspace(0, max_disp - 1, max_disp) # [0, 1, 2, ... , max_disp-1]
    index = index.to(disp_gt.device)
    index = index.repeat(b, h, w, 1).permute(0, 3, 1, 2).contiguous()

    probability = torch.abs(index - disp_gt)
    prob_mask = probability < 1
    probability = (1 - probability) * prob_mask
    probability = probability + 1e-40

    assert not torch.any(torch.isnan(probability))

    return probability
