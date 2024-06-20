# @Time    : 2024/6/20 20:44
# @Author  : zhangchenming
import torch


def get_model_params(model):
    return [p for p in model.parameters() if p.requires_grad]


class ClipGradValue:
    def __init__(self, clip_value):
        self.clip_value = clip_value

    def __call__(self, model):
        torch.nn.utils.clip_grad_value_(model.parameters(), self.clip_value)


class ClipGradNorm:
    def __init__(self, max_norm, norm_type=2):
        self.max_norm = max_norm
        self.norm_type = norm_type

    def __call__(self, model):
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm, self.norm_type)
