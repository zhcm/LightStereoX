# @Time    : 2024/6/9 18:13
# @Author  : zhangchenming
import os
from stereo.config.lazy import LazyCall
from stereo.datasets.speedbump import SpeedBump
from stereo.datasets.utils import stereo_trans
from stereo.datasets import build_dataloader

from cfgs.common.runtime_params import data_root_dir
from cfgs.common.constants import constants

data_root_path = os.path.join(data_root_dir, 'CarlaSpeedbumps')

trainv2 = LazyCall(SpeedBump)(
    data_root_path=data_root_path,
    split_file='/mnt/data/home/ruilin.wang/tools/speedbump/v2/train.txt',
    augmentations=None
)

trainv3 = LazyCall(SpeedBump)(
    data_root_path=data_root_path,
    split_file='/mnt/nas/public_data/stereo/StereoRBHM/CarlaSpeedbumpv3/train_new.txt',
    augmentations=None
)

val = LazyCall(SpeedBump)(
    data_root_path=data_root_path,
    # split_file='/mnt/data/home/ruilin.wang/tools/speedbump/v2/val.txt',
    split_file='/mnt/nas/public_data/stereo/StereoRBHM/CarlaSpeedbumpv3/val_new.txt',
    # split_file='/mnt/data/home/ruilin.wang/tools/speedbump/val_bisenet.txt',
    augmentations=[
        LazyCall(stereo_trans.ConstantPad)(target_size=[544, 960]),
        LazyCall(stereo_trans.NormalizeImage)(mean=constants.imagenet_rgb_mean, std=constants.imagenet_rgb_std)
    ]
)

val_loader = LazyCall(build_dataloader)(
    all_dataset=[val],
    batch_size=1,
    shuffle=False,
    workers=8,
    pin_memory=True)

