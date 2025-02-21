# @Time    : 2025/2/16 09:17
# @Author  : zhangchenming

def get_model_params(model):
    for name, param in model.named_parameters():
        if any([key in name for key in ['raft']]):
            param.requires_grad_(False)
    return model.parameters()


def get_all_model_params(model):
    return model.parameters()
