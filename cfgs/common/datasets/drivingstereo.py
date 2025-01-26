# @Time    : 2024/6/9 18:13
# @Author  : zhangchenming
import os
from stereo.config.lazy import LazyCall
from stereo.datasets.driving_dataset import DrivingDataset
from stereo.datasets.utils import stereo_trans
from stereo.datasets import build_dataloader

from cfgs.common.runtime_params import data_root_dir
from cfgs.common.constants import constants

data_root_path = os.path.join(data_root_dir, 'drivingstereo')

train = LazyCall(DrivingDataset)(
    data_root_path=data_root_path,
    split_file='./data/DrivingStereo/driving_stereo_train_174437.txt',
    augmentations=None
)

val = LazyCall(DrivingDataset)(
    data_root_path=data_root_path,
    split_file='./data/DrivingStereo/driving_stereo_full_test_7751.txt',
    augmentations=[
        LazyCall(stereo_trans.CropOrPad)(size=[800, 1760]),
        LazyCall(stereo_trans.NormalizeImage)(mean=constants.imagenet_rgb_mean, std=constants.imagenet_rgb_std)
    ]
)

val_loader = LazyCall(build_dataloader)(
    all_dataset=[val],
    batch_size=1,
    workers=8,
    pin_memory=True,
    shuffle=False)

# train (400, 881, 3]), (400, 879, 3)
# val full (800, 1758, 3), (800, 1762, 3)
