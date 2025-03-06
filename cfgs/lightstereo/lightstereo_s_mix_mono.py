# @Time    : 2024/10/8 02:37
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

train_augmentations = [
    LazyCall(stereo_trans.StereoColorJitter)(brightness=[0.6, 1.4], contrast=[0.6, 1.4],
                                             saturation=[0.6, 1.4], hue=[-0.5 / 3.14, 0.5 / 3.14],
                                             asymmetric_prob=0.2),
    LazyCall(stereo_trans.RandomErase)(prob=0.5, max_time=2, bounds=[50, 100]),
    LazyCall(stereo_trans.RandomCrop)(crop_size=[352, 640]),
    LazyCall(stereo_trans.NormalizeImage)(mean=constants.imagenet_rgb_mean, std=constants.imagenet_rgb_std)
]

carla = LazyConfig.load('cfgs/common/datasets/carla.py')  # 552050
carla.train.augmentations = train_augmentations
carla.weather_train.augmentations = train_augmentations

dynamic = LazyConfig.load('cfgs/common/datasets/dynamic.py')  # 144900
dynamic.train.augmentations = train_augmentations

crestereo = LazyConfig.load('cfgs/common/datasets/crestereo.py')  # 200000
crestereo.train.augmentations = train_augmentations
crestereo.train.return_right_disp = False

fallingthings = LazyConfig.load('cfgs/common/datasets/fallingthings.py')  # 61500
fallingthings.train.augmentations = train_augmentations
fallingthings.train.return_right_disp = False

instereo2k = LazyConfig.load('cfgs/common/datasets/instereo2k.py')  # 2010
instereo2k.train.augmentations = train_augmentations
instereo2k.train.return_right_disp = False

tartanair = LazyConfig.load('cfgs/common/datasets/tartanair.py')  # 306637
tartanair.train.augmentations = train_augmentations

sintel = LazyConfig.load('cfgs/common/datasets/sintel.py')  # 1064
sintel.train.augmentations = train_augmentations

spring = LazyConfig.load('cfgs/common/datasets/spring.py')  # 5000
spring.train.augmentations = train_augmentations
spring.train.return_right_disp = False

virtualkitti2 = LazyConfig.load('cfgs/common/datasets/virtualkitti2.py')  # 21260
virtualkitti2.train.augmentations = train_augmentations
virtualkitti2.train.return_right_disp = False

mono = LazyConfig.load('cfgs/common/datasets/mono.py')
mono.train_objects365_realfill.augmentations = train_augmentations

# dataloader
batch_size_per_gpu = 24
train_loader = LazyCall(build_dataloader)(
    is_dist=None,
    all_dataset=[tartanair.train, carla.train, carla.weather_train, crestereo.train, spring.train, sintel.train, dynamic.train, fallingthings.train, instereo2k.train, virtualkitti2.train, mono.train_objects365_realfill],
    batch_size=batch_size_per_gpu,
    shuffle=True,
    workers=8,
    pin_memory=True,
    drop_last=True)

val_loader = LazyCall(build_dataloader)(
    is_dist=None,
    all_dataset=[sintel.train],
    batch_size=batch_size_per_gpu * 2,
    shuffle=False,
    workers=8,
    pin_memory=True,
    drop_last=False)

# model
model = LazyCall(LightStereo)(
    backbone=LazyCall(MobileNetV2)(),
    max_disp=192,
    aggregation_blocks=[1, 2, 4],
    expanse_ratio=4,
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

runtime_params.save_root_dir = os.path.join(ckpt_root_dir, 'output/MixDataset/LightStereo_S')
runtime_params.train_epochs = 1
runtime_params.eval_period = 10
runtime_params.pretrained_model = os.path.join(ckpt_root_dir, 'output/SceneFlowDataset/LightStereo_S/cesc/ckpt/epoch_89/pytorch_model.bin')
