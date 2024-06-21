# @Time    : 2023/11/9 17:19
# @Author  : zhangchenming
import torch.nn.functional as F


def context_upsample(disp_low, up_weights, scale_factor=4):
    # disp_low [b,1,h,w]
    # up_weights [b,9,4*h,4*w]
    b, c, h, w = disp_low.shape
    disp_unfold = F.unfold(disp_low, kernel_size=3, dilation=1, padding=1)  # [bz, 3x3, hxw]
    disp_unfold = disp_unfold.reshape(b, -1, h, w)  # [bz, 3x3, h, w]
    disp_unfold = F.interpolate(disp_unfold, (h * scale_factor, w * scale_factor), mode='nearest')  # [bz, 3x3, 4h, 4w]
    disp = (disp_unfold * up_weights).sum(1)  # # [bz, 4h, 4w]

    return disp
