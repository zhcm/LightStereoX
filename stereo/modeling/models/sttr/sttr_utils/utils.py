# @Time    : 2024/1/16 19:16
# @Author  : zhangchenming
import torch


def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]


def torch_1d_sample(source, sample_points, mode='linear'):
    """
    linearly sample source tensor along the last dimension
    input:
        source [N,D1,D2,D3...,Dn]
        sample_points [N,D1,D2,....,Dn-1,1]
    output:
        [N,D1,D2...,Dn-1]
    """
    idx_l = torch.floor(sample_points).long().clamp(0, source.size(-1) - 1)
    idx_r = torch.ceil(sample_points).long().clamp(0, source.size(-1) - 1)

    if mode == 'linear':
        weight_r = sample_points - idx_l
        weight_l = 1 - weight_r
    elif mode == 'sum':
        weight_r = (idx_r != idx_l).int()  # we only sum places of non-integer locations
        weight_l = 1
    else:
        raise Exception('mode not recognized')

    out = torch.gather(source, -1, idx_l) * weight_l + torch.gather(source, -1, idx_r) * weight_r
    return out.squeeze(-1)
