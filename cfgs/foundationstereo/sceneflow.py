# @Time    : 2024/6/9 12:32
# @Author  : zhangchenming
import os
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from stereo.config.lazy import LazyCall, LazyConfig
from stereo.datasets import build_dataloader
from stereo.datasets.utils import stereo_trans
from stereo.modeling.models.foundationstereo.core.foundation_stereo import FoundationStereo
from stereo.solver.build import get_model_params, ClipGradValue

from cfgs.common.runtime_params import runtime_params, project_root_dir
from cfgs.common.constants import constants
from omegaconf import OmegaConf

# dataset
train_augmentations = [
    LazyCall(stereo_trans.StereoColorJitter)(brightness=[0.6, 1.4], contrast=[0.6, 1.4],
                                             saturation=[0.6, 1.4], hue=[-0.5/3.14, 0.5/3.14],
                                             asymmetric_prob=0.2),
    LazyCall(stereo_trans.RandomErase)(prob=0.5, max_time=2, bounds=[50, 100]),
    LazyCall(stereo_trans.RandomScale)(crop_size=[320, 736], min_pow_scale=-0.2, max_pow_scale=0.4,
                                       scale_prob=0.8, stretch_prob=0.8),
    LazyCall(stereo_trans.RandomCrop)(crop_size=[320, 736]),
    LazyCall(stereo_trans.NormalizeImage)(mean=constants.imagenet_rgb_mean, std=constants.imagenet_rgb_std)
]

sceneflow = LazyConfig.load('cfgs/common/datasets/sceneflow.py')
sceneflow.train.augmentations = train_augmentations

# dataloader
batch_size_per_gpu = 2
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
    batch_size=batch_size_per_gpu * 2,
    shuffle=False,
    workers=8,
    pin_memory=True)

# model
model = LazyCall(FoundationStereo)(
    args=OmegaConf.create(
        dict(
            corr_levels=2,
            corr_radius=4,
            hidden_dims=[128, 128, 128],
            low_memory=0,
            max_disp=416,
            mixed_precision=True,
            n_downsample=2,
            n_gru_layers=3,
            vit_size='vitl',
            train_iters=22,
            valid_iters=32
        ))
)

# optim
lr = 0.0001 * batch_size_per_gpu
optimizer = LazyCall(AdamW)(
    params=LazyCall(get_model_params)(model=None),
    lr=lr,
    weight_decay=1.0e-05,
    eps=1.0e-08)

# scheduler
scheduler = LazyCall(OneCycleLR)(optimizer=None, max_lr=lr, total_steps=-1, pct_start=0.01)

# clip grad
clip_grad = LazyCall(ClipGradValue)(clip_value=1.0)

# runtime params
runtime_params.save_root_dir = os.path.join(project_root_dir, 'output/SceneFlowDataset/FoundationStereo')
runtime_params.train_epochs = 90
runtime_params.mixed_precision = True
runtime_params.find_unused_parameters = True
