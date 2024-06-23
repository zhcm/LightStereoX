# @Time    : 2024/6/9 12:32
# @Author  : zhangchenming
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from stereo.config.lazy import LazyCall, LazyConfig
from stereo.datasets import build_dataloader
from stereo.datasets.utils import stereo_trans
from stereo.modeling.models.lightfast.lightstereo import LightStereo
from stereo.solver.build import get_model_params, ClipGradValue

from cfgs.common.runtime_params import runtime_params
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

driving = LazyConfig.load('cfgs/common/datasets/driving.py')
driving.train.augmentations = augmentations

# dataloader
batch_size_per_gpu = 6
train_loader = LazyCall(build_dataloader)(
    is_dist=None,
    all_dataset=[kitti12.trainval, kitti15.trainval, sceneflow.train, driving.train],
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
    max_disp=192,
    aggregation_blocks=[8, 16, 32],
    expanse_ratio=8,
    left_att=True,
    bigbk=True)

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
runtime_params.save_root_dir = ('/mnt/nas/algorithm/chenming.zhang/code/LightStereoX/output/'
                                'MultiDataset/LightStereo_Plus')
runtime_params.train_epochs = 90
runtime_params.mixed_precision = True
runtime_params.use_sync_bn = False
runtime_params.pretrained_model = ('/mnt/nas/algorithm/chenming.zhang/code/LightStereo/output/'
                                   'SceneFlowDataset/LightFast/lightstereo/fn-v2-b81632-e8-o-leftatt-efficientnet-lr6-epoch90-cesc/'
                                   'ckpt/checkpoint_epoch_89.pth')
