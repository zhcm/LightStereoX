# @Time    : 2024/6/9 12:32
# @Author  : zhangchenming
import os
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from stereo.config.lazy import LazyCall, LazyConfig
from stereo.datasets import build_dataloader
from stereo.datasets.utils import stereo_trans
from stereo.modeling.backbones.mobilenet import MobileNetV2
from stereo.modeling.models.stereobase.stereobase_gru import StereoBase
from stereo.solver.build import get_model_params, ClipGradValue

from cfgs.common.runtime_params import runtime_params, project_root_dir, ckpt_root_dir
from cfgs.common.constants import constants

# dataset
train_augmentations = [
    LazyCall(stereo_trans.RandomCrop)(crop_size=[320, 736]),
    LazyCall(stereo_trans.NormalizeImage)(mean=constants.rgb_mean, std=constants.rgb_std)
]

sceneflow = LazyConfig.load('cfgs/common/datasets/sceneflow.py')
sceneflow.train.augmentations = train_augmentations
sceneflow.train.return_right_disp = False

# dataloader
batch_size_per_gpu = 3
train_loader = LazyCall(build_dataloader)(
    is_dist=None,
    all_dataset=[sceneflow.train],
    batch_size=batch_size_per_gpu,
    shuffle=True,
    workers=8,
    pin_memory=True)

val_loader = LazyCall(build_dataloader)(
    is_dist=None,
    all_dataset=[sceneflow.val],
    batch_size=batch_size_per_gpu,
    shuffle=False,
    workers=8,
    pin_memory=True)

# model
model = LazyCall(StereoBase)(
    backbone=LazyCall(MobileNetV2)(),
    max_disp=192,
    num_groups=8,
    concat_channels=8,
    context_dims=[128, 128, 128],
    n_downsample=2,
    n_gru_layers=3,
    corr_radius=4,
    corr_levels=2,
    slow_fast_gru=False,
    train_iters=22,
    eval_iters=32
)

# optim
lr = 0.0001 * batch_size_per_gpu
optimizer = LazyCall(AdamW)(
    params=LazyCall(get_model_params)(model=None),
    lr=lr,
    weight_decay=1.0e-05,
    eps=1.0e-08)

# scheduler
scheduler = LazyCall(OneCycleLR)(optimizer=None, max_lr=lr, total_steps=-1, pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

# clip grad
clip_grad = LazyCall(ClipGradValue)(clip_value=1.0)

# runtime params
runtime_params.save_root_dir = os.path.join(ckpt_root_dir, 'output/SceneFlowDataset/StereoBase')
runtime_params.train_epochs = 90
runtime_params.mixed_precision = True
# runtime_params.freeze_bn = True
