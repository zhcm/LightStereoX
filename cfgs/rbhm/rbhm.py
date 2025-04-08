# @Time    : 2024/6/9 12:32
# @Author  : zhangchenming
import os
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from stereo.config.lazy import LazyCall, LazyConfig
from stereo.datasets import build_dataloader
from stereo.datasets.utils import stereo_trans
from stereo.modeling.models.coex.rbhm import HeightPred
from stereo.solver.build import get_model_params, ClipGradValue
from stereo.solver.trainer_rbhm import RBHMTrainer

from cfgs.common.runtime_params import runtime_params, project_root_dir
from cfgs.common.constants import constants

# dataset
train_augmentations = [
    LazyCall(stereo_trans.ConstantPad)(target_size=[544, 960]),
    LazyCall(stereo_trans.RandomCrop)(crop_size=[544, 736]),
    LazyCall(stereo_trans.NormalizeImage)(mean=constants.imagenet_rgb_mean, std=constants.imagenet_rgb_std)
]

speedbump = LazyConfig.load('cfgs/common/datasets/speedbump.py')
speedbump.trainv1.augmentations = train_augmentations
speedbump.trainv2.augmentations = train_augmentations
speedbump.trainv3.augmentations = train_augmentations
speedbump.trainv4.augmentations = train_augmentations

# dataloader
batch_size_per_gpu = 8
train_loader = LazyCall(build_dataloader)(
    is_dist=None,
    all_dataset=[speedbump.trainv4],
    batch_size=batch_size_per_gpu,
    shuffle=True,
    workers=8,
    pin_memory=True)

val_loader = LazyCall(build_dataloader)(
    is_dist=None,
    all_dataset=[speedbump.valv4],
    batch_size=batch_size_per_gpu,
    shuffle=False,
    workers=8,
    pin_memory=True)

# model
model = LazyCall(HeightPred)()

# optim
lr = 0.0001 * batch_size_per_gpu / 2
optimizer = LazyCall(AdamW)(
    params=LazyCall(get_model_params)(model=None),
    lr=lr,
    weight_decay=1.0e-05,
    eps=1.0e-08)

# scheduler
scheduler = LazyCall(OneCycleLR)(optimizer=None, max_lr=lr, total_steps=-1, pct_start=0.01)

# clip grad
clip_grad = LazyCall(ClipGradValue)(clip_value=0.1)

trainer = LazyCall(RBHMTrainer)(args=None, cfg=None, logger=None, tb_writer=None)

# runtime params
runtime_params.save_root_dir = os.path.join(project_root_dir, 'output/SpeedBumpDataset/RBHM')
runtime_params.train_epochs = 60
runtime_params.mixed_precision = False
