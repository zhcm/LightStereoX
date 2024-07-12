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
augmentations = [
    LazyCall(stereo_trans.RandomCrop)(crop_size=[320, 736]),
    LazyCall(stereo_trans.NormalizeImage)(mean=constants.imagenet_rgb_mean, std=constants.imagenet_rgb_std)
]

kitti12 = LazyConfig.load('cfgs/common/datasets/kitti12.py')
kitti12.trainval.augmentations = augmentations
kitti12.trainval.return_right_disp = False

kitti15 = LazyConfig.load('cfgs/common/datasets/kitti15.py')
kitti15.trainval.augmentations = augmentations
kitti15.trainval.return_right_disp = False

sceneflow = LazyConfig.load('cfgs/common/datasets/sceneflow.py')
sceneflow.train.augmentations = augmentations
sceneflow.train.return_right_disp = False

drivingstereo = LazyConfig.load('cfgs/common/datasets/drivingstereo.py')
drivingstereo.train.augmentations = augmentations

# dataloader
batch_size_per_gpu = 6
train_loader = LazyCall(build_dataloader)(
    is_dist=None,
    all_dataset=[kitti12.trainval, kitti15.trainval, sceneflow.train, drivingstereo.train],
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

# train params
runtime_params.save_root_dir = os.path.join(project_root_dir, 'output/MultiDataset/LightStereo_E')
runtime_params.train_epochs = 90
runtime_params.mixed_precision = True
runtime_params.use_sync_bn = False
runtime_params.pretrained_model = os.path.join(project_root_dir, 'output/SceneFlowDataset/LightStereo_E/cesc/ckpt/epoch_89/pytorch_model.bin')
