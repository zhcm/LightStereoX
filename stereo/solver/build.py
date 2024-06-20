# @Time    : 2024/6/20 20:44
# @Author  : zhangchenming
def get_model_params(model):
    return [p for p in model.parameters() if p.requires_grad]
