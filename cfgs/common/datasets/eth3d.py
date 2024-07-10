# @Time    : 2024/6/22 00:56
# @Author  : zhangchenming
import os
from stereo.config.lazy import LazyCall
from stereo.datasets.eth3d_dataset import ETH3DDataset
from stereo.datasets.utils import stereo_trans
from stereo.datasets import build_dataloader

from cfgs.common.runtime_params import data_root_dir
from cfgs.common.constants import constants

data_root_path = os.path.join(data_root_dir, 'ETH3D')

trainval = LazyCall(ETH3DDataset)(
    data_root_path=data_root_path,
    split_file='./data/ETH3D/eth3d_train.txt',
    augmentations=[
        LazyCall(stereo_trans.DivisiblePad)(divisor=32, mode='round'),
        LazyCall(stereo_trans.NormalizeImage)(mean=constants.imagenet_rgb_mean, std=constants.imagenet_rgb_std)
    ],
)

# 泛化测试，使用所有训练数据进行eval
val_loader = LazyCall(build_dataloader)(
    all_dataset=[trainval],
    batch_size=1,
    workers=8,
    pin_memory=True,
    shuffle=False)
