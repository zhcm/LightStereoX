# @Time    : 2024/6/9 12:32
# @Author  : zhangchenming
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from stereo.config.lazy import LazyCall, LazyConfig
from stereo.datasets import build_dataloader
from stereo.datasets.utils import stereo_trans
from stereo.datasets.utils import check
from stereo.modeling.models.lightfast.lightstereo import LightStereo
from stereo.solver.build import get_model_params, ClipGradValue

from cfgs.common.train_params import train_params
from cfgs.common.constants import constants


# dataset
check_augmentations = [
    LazyCall(stereo_trans.RandomCrop)(crop_size=[320, 736]),
    LazyCall(check.TransposeImage)(),
    LazyCall(check.ToTensor)(),
    LazyCall(check.NormalizeImage)(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
augmentations = [
    LazyCall(stereo_trans.RandomCrop)(crop_size=[320, 736]),
    LazyCall(stereo_trans.NormalizeImage)(mean=constants.imagenet_rgb_mean, std=constants.imagenet_rgb_std)
]

kitti12 = LazyConfig.load('cfgs/common/datasets/kitti12.py')
kitti12.trainval.augmentations = check_augmentations
kitti12.trainval.return_right_disp = False

kitti15 = LazyConfig.load('cfgs/common/datasets/kitti15.py')
kitti15.trainval.augmentations = check_augmentations
kitti15.trainval.return_right_disp = False

sceneflow = LazyConfig.load('cfgs/common/datasets/sceneflow.py')
sceneflow.train.augmentations = check_augmentations
sceneflow.train.return_right_disp = False

driving = LazyConfig.load('cfgs/common/datasets/driving.py')
driving.train.augmentations = check_augmentations

# dataloader
batch_size_per_gpu = 12
train_loader = LazyCall(build_dataloader)(
    is_dist=None,
    all_dataset=[kitti12.trainval, kitti15.trainval, sceneflow.train, driving.train],
    batch_size=batch_size_per_gpu,
    shuffle=True,
    workers=8,
    pin_memory=True)

val_loader = LazyCall(build_dataloader)(
    is_dist=None,
    all_dataset=[kitti15.trainval],
    batch_size=batch_size_per_gpu,
    shuffle=False,
    workers=8,
    pin_memory=True)

# model
model = LazyCall(LightStereo)(
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

# eval params
eval_params = dict(
    eval_max_disp=192
)

# train params
train_params.save_root_dir = ('/mnt/nas/algorithm/chenming.zhang/code/LightStereoX/output/'
                              'MultiDataset/LightStereo_S')
train_params.train_epochs = 90
train_params.mixed_precision = True
