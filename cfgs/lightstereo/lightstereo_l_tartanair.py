# @Time    : 2024/6/9 12:32
# @Author  : zhangchenming
import os
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from stereo.config.lazy import LazyCall, LazyConfig
from stereo.datasets import build_dataloader
from stereo.datasets.utils import stereo_trans
from stereo.modeling.backbones.mobilenet import MobileNetV2
from stereo.modeling.models.lightstereo.lightstereo import LightStereo
from stereo.solver.build import get_model_params, ClipGradValue

from cfgs.common.runtime_params import runtime_params, ckpt_root_dir
from cfgs.common.constants import constants

# dataset
train_augmentations = [
    LazyCall(stereo_trans.StereoColorJitter)(brightness=[0.6, 1.4], contrast=[0.6, 1.4],
                                             saturation=[0.6, 1.4], hue=[-0.15, 0.15],
                                             asymmetric_prob=0.2),
    LazyCall(stereo_trans.RandomErase)(prob=0.5, max_time=2, bounds=[50, 100]),
    LazyCall(stereo_trans.RandomScale)(crop_size=[320, 736], min_pow_scale=-0.2, max_pow_scale=0.4,
                                       scale_prob=0.8, stretch_prob=0.8),
    LazyCall(stereo_trans.RandomCrop)(crop_size=[320, 736]),
    LazyCall(stereo_trans.NormalizeImage)(mean=constants.imagenet_rgb_mean, std=constants.imagenet_rgb_std)
]

data = LazyConfig.load('cfgs/common/datasets/tartanair.py')
data.train.augmentations = train_augmentations

# dataloader
batch_size_per_gpu = 8
train_loader = LazyCall(build_dataloader)(
    is_dist=None,
    all_dataset=[data.train],
    batch_size=batch_size_per_gpu,
    shuffle=True,
    workers=8,
    pin_memory=True)

val_loader = LazyCall(build_dataloader)(
    is_dist=None,
    all_dataset=[data.train],
    batch_size=batch_size_per_gpu,
    shuffle=False,
    workers=8,
    pin_memory=True)

# model
model = LazyCall(LightStereo)(
    backbone=LazyCall(MobileNetV2)(),
    max_disp=192,
    aggregation_blocks=[8, 16, 32],
    expanse_ratio=8,
    left_att=True)

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
clip_grad = LazyCall(ClipGradValue)(clip_value=0.1)

# runtime params
runtime_params.save_root_dir = os.path.join(ckpt_root_dir, 'output/TartanAirDataset/LightStereo_L')
runtime_params.max_iter = int(700000 / 8 / 8)
runtime_params.eval_period = 10
runtime_params.mixed_precision = False
runtime_params.pretrained_model = os.path.join(ckpt_root_dir, 'output/SceneFlowDataset/LightStereo_L/cesc/ckpt/epoch_89/pytorch_model.bin')
