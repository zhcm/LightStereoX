# @Time    : 2024/6/9 12:32
# @Author  : zhangchenming
import os
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from stereo.config.lazy import LazyCall, LazyConfig
from stereo.datasets import build_dataloader
from stereo.datasets.utils import stereo_trans
from stereo.modeling.backbones.efficientnetv2 import EfficientNetV2
from stereo.modeling.models.lightstereo.lightstereo import LightStereo
from stereo.solver.build import get_model_params, ClipGradValue

from cfgs.common.runtime_params import runtime_params, project_root_dir
from cfgs.common.constants import constants

# dataset
train_augmentations = [
    LazyCall(stereo_trans.RandomCrop)(crop_size=[320, 736]),
    LazyCall(stereo_trans.NormalizeImage)(mean=constants.imagenet_rgb_mean, std=constants.imagenet_rgb_std)
]

fallingthings = LazyConfig.load('cfgs/common/datasets/fallingthings.py')  # (540, 960, 3)
fallingthings.train.augmentations = train_augmentations
fallingthings.train.return_right_disp = False

instereo2k = LazyConfig.load('cfgs/common/datasets/instereo2k.py')  # (860, 1080, 3)
instereo2k.train.augmentations = train_augmentations
instereo2k.train.return_right_disp = False

sceneflow = LazyConfig.load('cfgs/common/datasets/sceneflow.py')  # (540, 960, 3)
sceneflow.train.augmentations = train_augmentations
sceneflow.train.return_right_disp = False

sintel = LazyConfig.load('cfgs/common/datasets/sintel.py')  # (436, 1024, 3)
sintel.train.augmentations = train_augmentations

unrealstereo4k = LazyConfig.load('cfgs/common/datasets/unrealstereo4k.py')  # (2160, 3840, 3)
unrealstereo4k.train.augmentations = train_augmentations
unrealstereo4k.train.return_right_disp = False

virtualkitti2 = LazyConfig.load('cfgs/common/datasets/virtualkitti2.py')  # (375, 1242, 3)
virtualkitti2.train.augmentations = train_augmentations
virtualkitti2.train.return_right_disp = False

# dataloader
batch_size_per_gpu = 6
train_loader = LazyCall(build_dataloader)(
    is_dist=None,
    all_dataset=[fallingthings.train, instereo2k.train, sceneflow.train,  sintel.train, unrealstereo4k.train, virtualkitti2.train],
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
model = LazyCall(LightStereo)(
    backbone=LazyCall(EfficientNetV2)(),
    max_disp=192,
    aggregation_blocks=[8, 16, 32],
    expanse_ratio=8,
    left_att=True)

# optim
lr = 0.0002 * batch_size_per_gpu
optimizer = LazyCall(AdamW)(
    params=LazyCall(get_model_params)(model=None),
    lr=lr,
    weight_decay=1.0e-05,
    eps=1.0e-08)

# scheduler
scheduler = LazyCall(OneCycleLR)(optimizer=None, max_lr=lr, total_steps=-1, pct_start=0.01)

# clip grad
clip_grad = LazyCall(ClipGradValue)(clip_value=0.1)

# train params
runtime_params.save_root_dir = os.path.join(project_root_dir, 'output/MultiDataset/LightStereo_E')
runtime_params.train_epochs = 30
runtime_params.mixed_precision = True
runtime_params.pretrained_model = os.path.join(project_root_dir, 'output/SceneFlowDataset/LightStereo_E/cesc/ckpt/epoch_89/pytorch_model.bin')
