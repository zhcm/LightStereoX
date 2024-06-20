# @Time    : 2023/11/9 17:41
# @Author  : zhangchenming
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)


def topkpool_regression(cost, k):
    """
    :param cost: # [bz, max_disp, h, w]
    :param k:
    :return:
    """
    _, ind = cost.sort(dim=1, descending=True)
    pool_ind = ind[:, :k, :, :]
    cv = torch.gather(cost, dim=1, index=pool_ind)
    prob = F.softmax(cv, dim=1)
    disp = torch.sum(prob * pool_ind, dim=1, keepdim=True)
    return disp
