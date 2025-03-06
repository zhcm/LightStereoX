# @Time    : 2024/6/9 12:32
# @Author  : zhangchenming
import os
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from stereo.config.lazy import LazyCall, LazyConfig
from stereo.datasets import build_dataloader
from stereo.datasets.utils import stereo_trans
from stereo.modeling.models.igev.igev_stereo_speedbump import IGEVStereo
from stereo.solver.build import get_model_params, ClipGradValue, ClipGradNorm

from cfgs.common.runtime_params import runtime_params, project_root_dir
from cfgs.common.constants import constants

# dataset
train_augmentations = [
    LazyCall(stereo_trans.StereoColorJitter)(brightness=[0.6, 1.4], contrast=[0.6, 1.4],
                                             saturation=[0.6, 1.4], hue=[-0.5 / 3.14, 0.5 / 3.14],
                                             asymmetric_prob=0.2),
    LazyCall(stereo_trans.RandomErase)(prob=0.5, max_time=2, bounds=[50, 100]),
    LazyCall(stereo_trans.ConstantPad)(target_size=[544, 960]),
    LazyCall(stereo_trans.RandomCrop)(crop_size=[544, 736]),
    LazyCall(stereo_trans.NormalizeImage)(mean=constants.imagenet_rgb_mean, std=constants.imagenet_rgb_std)
]

speedbump = LazyConfig.load('cfgs/common/datasets/speedbump.py')
speedbump.trainv3.augmentations = train_augmentations

# dataloader
batch_size_per_gpu = 1
train_loader = LazyCall(build_dataloader)(
    is_dist=None,
    all_dataset=[speedbump.trainv3],
    batch_size=batch_size_per_gpu,
    shuffle=True,
    workers=8,
    pin_memory=True)

val_loader = LazyCall(build_dataloader)(
    is_dist=None,
    all_dataset=[speedbump.val],
    batch_size=batch_size_per_gpu,
    shuffle=False,
    workers=8,
    pin_memory=True)

# model
model = LazyCall(IGEVStereo)()

# optim
lr = 0.0001
optimizer = LazyCall(AdamW)(
    params=LazyCall(get_model_params)(model=None),
    lr=lr,
    weight_decay=1.0e-05,
    eps=1.0e-08)

# scheduler
scheduler = LazyCall(OneCycleLR)(optimizer=None, max_lr=lr, total_steps=-1, pct_start=0.01)

# clip grad
clip_grad = LazyCall(ClipGradNorm)(max_norm=35)

# train params
runtime_params.save_root_dir = os.path.join(project_root_dir, 'output/SpeedBumpDataset/IGEV')
runtime_params.train_epochs = 20
runtime_params.mixed_precision = False
runtime_params.find_unused_parameters = True
runtime_params.pretrained_model = os.path.join(project_root_dir, 'ckpt/IGEV_Sceneflow_Amp.pt')

