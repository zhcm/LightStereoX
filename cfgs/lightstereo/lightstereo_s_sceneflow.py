# @Time    : 2024/6/9 12:32
# @Author  : zhangchenming
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from stereo.config.lazy import LazyCall, LazyConfig
from stereo.datasets import build_dataloader
from stereo.modeling.models.lightfast.lightstereo import LightStereo
from stereo.solver.build import get_model_params, ClipGradValue

from cfgs.common.train_params import train_params

# model
model = LazyCall(LightStereo)(
    max_disp=192,
    aggregation_blocks=[1, 2, 4],
    expanse_ratio=4,
    left_att=True)

# optim
lr = 0.0001 * 24
optimizer = LazyCall(AdamW)(
    params=LazyCall(get_model_params)(model=None),
    lr=lr,
    weight_decay=1.0e-05,
    eps=1.0e-08)

# scheduler
scheduler = LazyCall(OneCycleLR)(optimizer=None, max_lr=lr, total_steps=-1, pct_start=0.01)

# CLIP GRAD
clip_grad = LazyCall(ClipGradValue)(clip_value=0.1)


# train params
train_params.save_root_dir = ('/mnt/nas/algorithm/chenming.zhang/code/LightStereoX/output/'
                              'SceneFlowDataset/LightStereo_S')
train_params.train_epochs = 90
train_params.mixed_precision = True
