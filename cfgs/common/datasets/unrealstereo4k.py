# @Time    : 2024/6/9 18:13
# @Author  : zhangchenming
import os
from stereo.config.lazy import LazyCall
from stereo.datasets.unrealstereo4k_dataset import UnrealStereo4KDataset
from stereo.datasets.utils import stereo_trans
from stereo.datasets import build_dataloader

from cfgs.common.runtime_params import data_root_dir
from cfgs.common.constants import constants

data_root_path = os.path.join(data_root_dir, 'UnrealStereo4K')

train = LazyCall(UnrealStereo4KDataset)(
    data_root_path=data_root_path,
    split_file='./data/UnrealStereo4K/unrealstereo4k_train_7380.txt',
    augmentations=None,
    return_right_disp=True
)

val = LazyCall(UnrealStereo4KDataset)(
    data_root_path=data_root_path,
    split_file='./data/UnrealStereo4K/unrealstereo4k_val_820.txt',
    augmentations=[
        LazyCall(stereo_trans.DivisiblePad)(divisor=32, mode='round'),
        LazyCall(stereo_trans.NormalizeImage)(mean=constants.imagenet_rgb_mean, std=constants.imagenet_rgb_std)
    ],
    return_right_disp=False
)

val_loader = LazyCall(build_dataloader)(
    all_dataset=[val],
    batch_size=1,
    workers=8,
    pin_memory=True,
    shuffle=False)

# (2160, 3840, 3)
