# @Time    : 2024/6/9 18:13
# @Author  : zhangchenming
import os
from stereo.config.lazy import LazyCall
from stereo.datasets.instereo2k_dataset import InStereo2KDataset
from stereo.datasets.utils import stereo_trans
from stereo.datasets import build_dataloader

from cfgs.common.runtime_params import data_root_dir
from cfgs.common.constants import constants

data_root_path = os.path.join(data_root_dir, 'InStereo2K')

train = LazyCall(InStereo2KDataset)(
    data_root_path=data_root_path,
    split_file='./data/InStereo2K/instereo2k_train_2010.txt',
    augmentations=None,
    return_right_disp=True
)  # 2010

val = LazyCall(InStereo2KDataset)(
    data_root_path=data_root_path,
    split_file='./data/InStereo2K/instereo2k_test_50.txt',
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

# (860, 1080, 3)
