# @Time    : 2024/6/9 18:13
# @Author  : zhangchenming
import os
from stereo.config.lazy import LazyCall
from stereo.datasets.vkitti2_dataset import VirtualKitti2Dataset
from stereo.datasets.utils import stereo_trans
from stereo.datasets import build_dataloader

from cfgs.common.runtime_params import data_root_dir
from cfgs.common.constants import constants

data_root_path = os.path.join(data_root_dir, 'VirtualKITTI2')

train = LazyCall(VirtualKitti2Dataset)(
    data_root_path=data_root_path,
    split_file='./data/VirtualKitti2/virtualkitti2_train_19134.txt',
    augmentations=None,
    return_right_disp=True
)

val = LazyCall(VirtualKitti2Dataset)(
    data_root_path=data_root_path,
    split_file='./data/VirtualKitti2/virtualkitti2_val_2126.txt',
    augmentations=[
        LazyCall(stereo_trans.DivisiblePad)(divisor=32, mode='round'),
        LazyCall(stereo_trans.NormalizeImage)(mean=constants.imagenet_rgb_mean, std=constants.imagenet_rgb_std)
    ],
    return_right_disp=False
)

val_loader = LazyCall(build_dataloader)(
    all_dataset=[val],
    batch_size=2,
    workers=8,
    pin_memory=True,
    shuffle=False)

# (375, 1242, 3)
