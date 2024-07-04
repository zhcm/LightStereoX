# @Time    : 2024/7/3 01:37
# @Author  : zhangchenming
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from stereo.config.lazy import LazyCall, LazyConfig
from stereo.datasets import build_dataloader
from stereo.datasets.utils import stereo_trans
from stereo.modeling.models.sttr.sttr import STTR, get_model_params
from stereo.modeling.models.sttr.sttr_utils.backbone import SppBackbone
from stereo.modeling.models.sttr.sttr_utils.repvit import RepVit
from stereo.solver.build import ClipGradNorm
from stereo.solver.warmup import LinearWarmup

from cfgs.common.runtime_params import runtime_params
from cfgs.common.constants import constants

# dataset
train_augmentations = [
    LazyCall(stereo_trans.RandomCrop)(crop_size=[360, 640]),
    LazyCall(stereo_trans.NormalizeImage)(mean=constants.imagenet_rgb_mean, std=constants.imagenet_rgb_std)
]

flyingthings3dsubset = LazyConfig.load('cfgs/common/datasets/flyingthings3dsubset.py')
flyingthings3dsubset.train.augmentations = train_augmentations

# dataloader
batch_size_per_gpu = 1
train_loader = LazyCall(build_dataloader)(
    is_dist=None,
    all_dataset=[flyingthings3dsubset.train],
    batch_size=batch_size_per_gpu,
    shuffle=True,
    workers=8,
    pin_memory=True,
    batch_uniform=True,
    h_range=[1.0, 1.78],
    w_range=[1.0, 1.5])

val_loader = LazyCall(build_dataloader)(
    is_dist=None,
    all_dataset=[flyingthings3dsubset.val],
    batch_size=batch_size_per_gpu,
    shuffle=False,
    workers=8,
    pin_memory=True)

# model
model = LazyCall(STTR)(
    backbone=LazyCall(SppBackbone)()
)

# optim
lr = 0.0008
optimizer = LazyCall(Adam)(
    params=LazyCall(get_model_params)(model=None, base_lr=lr),
    lr=lr,
    betas=(0.9, 0.999))

# scheduler
scheduler = LazyCall(MultiStepLR)(optimizer=None, milestones=[5, 7, 9], gamma=0.5)

# warmup
warmup = LazyCall(LinearWarmup)(optimizer=None, warmup_period=2000, last_step=-1)

# clip grad
clip_grad = LazyCall(ClipGradNorm)(max_norm=35)

# runtime params
runtime_params.save_root_dir = ('/mnt/nas/algorithm/chenming.zhang/code/LightStereoX/output/'
                                'SceneFlowDataset/STTR')
runtime_params.train_epochs = 10
runtime_params.find_unused_parameters = False  # must be false, there is a bug with torch.utils.checkpoint and find_unused_parameters=True
