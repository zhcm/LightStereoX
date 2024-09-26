# @Time    : 2024/6/9 18:13
# @Author  : zhangchenming
import os
from stereo.config.lazy import LazyCall
from stereo.datasets.sintel_dataset import SintelDataset
from stereo.datasets.utils import stereo_trans
from stereo.datasets import build_dataloader

from cfgs.common.runtime_params import data_root_dir
from cfgs.common.constants import constants

data_root_path = os.path.join(data_root_dir, 'Sintel')

train = LazyCall(SintelDataset)(
    data_root_path=data_root_path,
    split_file='./data/Sintel/sintel_final_train_957.txt',
    augmentations=None
)

val = LazyCall(SintelDataset)(
    data_root_path=data_root_path,
    split_file='./data/Sintel/sintel_final_val_107.txt',
    augmentations=[
        LazyCall(stereo_trans.DivisiblePad)(divisor=32, mode='round'),
        LazyCall(stereo_trans.NormalizeImage)(mean=constants.imagenet_rgb_mean, std=constants.imagenet_rgb_std)
    ]
)

val_loader = LazyCall(build_dataloader)(
    all_dataset=[val],
    batch_size=2,
    workers=8,
    pin_memory=True,
    shuffle=False)

# (436, 1024, 3)
